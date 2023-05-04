# -*- coding: utf-8 -*-
import time
import os,sys,glob,copy,pickle
import torch
import torchvision
import torchvision.transforms as transforms
import random
from build_fg_cpu import build_data_strict_all,build_gkernel,\
                     build_data_smtry,build_data_strict_smtry,build_data_strict_paths
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import networks
from scipy import ndimage #new
import torch.optim as optim

def write_out_pdb(fpath,vecs,scores,scorecut = None):
    form = 'HETATM%5d  O   HOH X%4d    %8.3f%8.3f%8.3f  1.00%6.2f\n'
    length = len(vecs)
    f = open(fpath,'w')
    dump = []
    for i in range(length):
        write = False
        if (scorecut == None):
            write = True
        elif scores[i] > scorecut:
            write = True
        else:
            write = False
        if write:
            newline = form%((i+1),(1),vecs[i][0],vecs[i][1],vecs[i][2],scores[i])
            dump.append( [ scores[i] , newline ] )
    #sort!
    dump_sorted = sorted(dump, key = lambda x: -1.0*x[0])
    for i in range(len(dump_sorted)):
        f.write(dump_sorted[i][1])
    f.close()
def place_water(prob, null, vec_start, kernel, vcut=1.0, grid =0.5,padding=4.0,ignore_padding=False):
    #prob, ban: 3 dimension array
    #probability: result from net
    #null: True when atom(C,N,O, water) NOT exists 
    #vec_start: coord of prob[0][0][0]

    #build oxygen sized gaussian kernel
    #1.52: radius of oxygen atom (in angstrom) 
    r_grid = int( 1 + (1.5*1.52)/grid ) #radius in grid debug
    d_grid = 2*r_grid+1
    result = [] 
    scores = [] 
    maxv = 999.999
   
    prob_new = prob * null
    cv  = ndimage.convolve(prob_new, kernel, mode='reflect')

    while (maxv >= vcut):
        #1. since kernel is symmetric for i operation -> ignore about convolution eqn
        #2. reflect was used for treating sliced water at border 
        
        ind = np.unravel_index(np.argmax(cv,axis=None),cv.shape) #sth like (20,30,40)
        vec = vec_start + grid*np.array(ind)
        maxv = cv[ind]

        if (maxv < vcut):
            break
       
        result.append(vec)
        scores.append(maxv)

        for v in range(d_grid**3):
            i = v % d_grid
            j = (v // d_grid)%d_grid
            k = (v // (d_grid*d_grid))%d_grid
            
            x = min(prob.shape[0]-1, max(0, i+ind[0]-r_grid)) 
            y = min(prob.shape[1]-1, max(0, j+ind[1]-r_grid))
            z = min(prob.shape[2]-1, max(0, k+ind[2]-r_grid))
            dx = grid*(i - r_grid) 
            dy = grid*(j - r_grid) 
            dz = grid*(k - r_grid) 

            d =  (dx**2 + dy**2 + dz**2)**(0.5)
            if d < (1.5*1.52):
               cv[x][y][z] = 0.0
    if ignore_padding:
        result_new = []
        scores_new = []
        for i,vec in enumerate(result):
            diff    = vec - vec_start
            n_grids = prob.shape
            ignored = False
            for j in range(len(prob.shape)):
                if diff[j] < padding:
                    ignored = True
                    break
                elif diff[j] > (grid*n_grids[j] - padding):
                    ignored = True
                    break
            if not ignored:
                result_new.append(vec)
                scores_new.append(scores[i])
        return result_new,scores_new
    else:
        return result,scores

def batch_idxs(idxs, batch=4, shuffle=True):
    result = []
    if shuffle:
        random.shuffle(idxs)
    len_idxs = len(idxs)
    for i in range(len_idxs//batch):
        result.append([idxs[batch*i+j] for j in range(batch)])
    return result

def build_batch(batch,train=True,n_grid=32,dbloss=False):
    inputs_l = []
    outputs_l = []
    w_l = []
    vec_start = []
    for idx in batch:
        if train:
            vs,inp,out,w = build_data_smtry(idx,n_grid=n_grid,dbloss=dbloss)
        else:
            vs,inp,out,w = build_data_strict_smtry(idx,n_grid=n_grid,dbloss=dbloss)
        inputs_l.append(inp)
        outputs_l.append(out)
        w_l.append(w)
        vec_start.append(vs)
    
    inputs  = np.vstack(inputs_l)
    outputs = np.vstack(outputs_l)
    ws = np.vstack(w_l)
    return vec_start,inputs,outputs,ws       

def build_batch_all(batch,padding=32.0,dbloss=False):
    inputs_l = []
    outputs_l = []
    w_l = []
    vec_start = []
    for idx in batch:
        vs,inp,out,w = build_data_strict_all(idx,padding=padding,dbloss=dbloss)
        inputs_l.append(inp)
        outputs_l.append(out)
        w_l.append(w)
        vec_start.append(vs)
    inputs  = np.vstack(inputs_l)
    outputs = np.vstack(outputs_l)
    ws = np.vstack(w_l)
    return vec_start,inputs,outputs,ws       

def build_batch_full_paths(paths,grid=0.5,n_grid=64,padding=4.0,dbloss=False):
    #'paths : {"pro":[pro_paths...],"wat":[wat_paths(=answer,for training, optional)] }
    inputs_l = []
    outputs_l = []
    w_l = []
    vec_start = []
    data_old = build_data_strict_paths(paths,grid=0.5,n_grid=64,padding=4.0,dbloss=dbloss)
    data = []
    for datum in data_old:
        vs  = np.array( [datum[0]] )
        inp = np.array( datum[1] )
        out = np.array( datum[2] )
        w   = np.array( datum[3] )
        data.append( (vs,inp,out,w))
    return data       

def save_grid(vec_starts, np_inputs, out, batch, dr ,vcut=0.6,grid=0.5 ):
    kernel = build_gkernel(grid=grid, r = 1.52)
    np_o    = np.expand_dims((np_inputs[:,0,:,:,:] > 0.5),axis=1)  
    np_c    = np.expand_dims((np_inputs[:,1,:,:,:] > 0.5),axis=1) 
    np_n    = np.expand_dims((np_inputs[:,2,:,:,:] > 0.5),axis=1) 
    np_s    = np.expand_dims((np_inputs[:,3,:,:,:] > 0.5),axis=1) 
    np_prot = np.logical_or(np.logical_or(np.logical_or(np_o,np_c),np_n),np_s)
    np_null = np.logical_not(np_prot)
    
    for trg_idx in range((out.shape[0])): 
        zipped = {'vec_start':vec_starts[trg_idx],
                  'wat_grid':out[trg_idx][0],
                  'wat_grid_noprot':(out[trg_idx][0] * np_null[trg_idx][0]),
                  'grid':grid}
        
        fpath = '%s/%s_grid.bin'%(dr,batch[trg_idx])
        grid_dl = open(fpath,'wb')
        pickle.dump(zipped,grid_dl)
def run_place_water(vec_starts, np_inputs, out, batch, dr ,vcut=0.6,grid=0.5 ):
    kernel = build_gkernel(grid=grid, r = 1.52) 
    np_o    = np.expand_dims((np_inputs[:,0,:,:,:] > 0.5),axis=1)  
    np_c    = np.expand_dims((np_inputs[:,1,:,:,:] > 0.5),axis=1) 
    np_n    = np.expand_dims((np_inputs[:,2,:,:,:] > 0.5),axis=1) 
    np_s    = np.expand_dims((np_inputs[:,3,:,:,:] > 0.5),axis=1) 
    np_prot = np.logical_or(np.logical_or(np.logical_or(np_o,np_c),np_n),np_s)
    np_null = np.logical_not(np_prot)
    
    for trg_idx in range((out.shape[0])): 
        watvecs,scores =  place_water(out[trg_idx][0], np_null[trg_idx][0], 
                                    vec_starts[trg_idx], kernel, vcut=vcut, grid =grid)
        zipped = {'vecs':watvecs, 'scores':scores}
        fpath = '%s/%s.pdb'%(dr,batch[trg_idx])
        write_out_pdb(fpath,watvecs,scores)

def run_place_water_part(vec_starts, np_inputs, out, vcut=0.6,grid=0.5,padding=4.0):
    kernel = build_gkernel(grid=grid, r = 1.52) 
    np_o    = np.expand_dims((np_inputs[:,0,:,:,:] > 0.5),axis=1)  
    np_c    = np.expand_dims((np_inputs[:,1,:,:,:] > 0.5),axis=1) 
    np_n    = np.expand_dims((np_inputs[:,2,:,:,:] > 0.5),axis=1) 
    np_s    = np.expand_dims((np_inputs[:,3,:,:,:] > 0.5),axis=1) 
    np_prot = np.logical_or(np.logical_or(np.logical_or(np_o,np_c),np_n),np_s)
    np_null = np.logical_not(np_prot)
   
    for trg_idx in range((out.shape[0])): 
        watvecs,scores =  place_water(out[trg_idx][0], np_null[trg_idx][0], 
                                    vec_starts[trg_idx], kernel, vcut=vcut, grid =grid,
                                    padding=padding,ignore_padding=True)
        
        is_bound = [False for i in range(len(watvecs))]
        for i,vec in enumerate(watvecs):
            diff    = vec - vec_starts[trg_idx]
            n_grids = out[trg_idx][0].shape
            for j in range(3):
                if diff[j] < (padding+2.25):
                    is_bound[i] = True
                    break
                elif diff[j] > (grid*n_grids[j] - padding-2.25):
                    is_bound[i] = True
                    break
        zipped = {'vecs':watvecs, 'scores':scores,'is_bound':is_bound}
    return zipped
def test_epoch(env, epoch, batchsize=4):
    with torch.no_grad():
        run_epoch(env, epoch, train=False, build=False, batchsize=batchsize)

def train_epoch(env, epoch, batchsize=4):
    run_epoch(env, epoch, train=True, build=False, batchsize=batchsize)

def build_epoch(env, epoch, train=True, prefix='build'):
    with torch.no_grad():
        run_epoch(env, epoch, train=train, build=True, batchsize=1 , dr='./')
def build_full_epoch(env, epoch, train=True, prefix='build',tlog=False,tlog_path='cpu_time.txt'):
    with torch.no_grad():
        run_full_epoch(env, epoch, train=train, dr='./',tlog=tlog,tlog_path='cpu_time.txt')

#run_full_epoch_start
def vecs_2_idxs(vecs):
    xs_temp = []
    ys_temp = []
    zs_temp = []
    for vec_start in vecs:
        xs_temp.append(vec_start[0])
        ys_temp.append(vec_start[1])
        zs_temp.append(vec_start[2])
    xs = list(set(xs_temp))
    ys = list(set(ys_temp))
    zs = list(set(zs_temp))
    xs.sort()
    ys.sort()
    zs.sort()
    return xs,ys,zs
def run_full_epoch(env, epoch, train=True,build=False, dr=None,tlog = False, tlog_path='cpu_time.txt'):
    idxs      = env['idxs']
    net       = env['net']
    device    = env['device']
    optimizer = env['optimizer']
    n_grid    = env['n_grid']
    padding   = env['padding']
    dbloss    = env['dbloss']
    if 'use_paths' not in env.keys():
        use_paths = False
    else:
        use_paths = env['use_paths']
    running_loss = 0.0
    total_loss = 0.0
    if tlog:
        time_log = open(tlog_path,'a')
        time_start = 0
        time_end = 0
    else:
        time_log = None
        time_start = 0
        time_end = 0

    for i, idx in enumerate(idxs):
        time_start = time.time()
        wat_dict = {'vecs':[] , 'scores':[],'is_bound':[]}
        if use_paths:
            paths = env['paths_dict'][idx]
            #vec_starts: [vec_start] (current build_batch_full_paths uses only one target for batch)
            data = build_batch_full_paths(paths,grid=0.5,n_grid=n_grid,padding=padding,dbloss=dbloss) 
        else:
            #vec_starts: [vec_start] (current build_batch_full uses only one target for batch)
            data = build_batch_full(idx,grid=0.5,n_grid=n_grid,padding=padding,dbloss=dbloss) 
       
        vec_starts_part = []
        for datum in data:
            #vs: [vec_start] (current build_batch_full uses only one target for batch)
            vs, np_inputs_part,np_answers_part,np_weights_part  = datum
            vec_starts_part.append(vs[0])
        np_inputs_part = data[0][1]
        xs,ys,zs =  vecs_2_idxs(vec_starts_part)
        len_idxs = [len(xs), len(ys), len(zs)]
        #grid_out: 2: doubleloss / 1: singleloss
        n_out_ch = 0
        if dbloss:
            n_out_ch = 2
        else:
            n_out_ch = 1
        grid_out =  np.zeros( (np_inputs_part.shape[0],n_out_ch,48*len(xs)+16 , 48*len(ys)+16 , 48*len(zs)+16 )) 
        grid_prot =  np.zeros( (np_inputs_part.shape[0], np_inputs_part.shape[1],48*len(xs)+16 , 48*len(ys)+16 , 48*len(zs)+16 )) 
        wat_dict = {'vecs':[] , 'scores':[],'is_bound':[]}
        
        for data_idx, datum in enumerate(data):
            #vec_starts: [vec_start] (current build_batch_full uses only one target for batch)
            vec_starts, np_inputs,np_answers,np_weights  = datum 

            inputs  = torch.tensor(torch.FloatTensor(np_inputs),  device=device ,requires_grad=False, dtype=torch.float32)
            answers = torch.tensor(torch.FloatTensor(np_answers), device=device ,requires_grad=False, dtype=torch.float32)
            weights = torch.tensor(torch.FloatTensor(np_weights), device=device ,requires_grad=False, dtype=torch.float32)
            inputs  = inputs.to(device=device) #gpu
            answers = answers.to(device=device) #gpu
            weights = weights.to(device=device) #gpu
    
            outputs = net.eval()(inputs) 
            lossf = nn.BCEWithLogitsLoss(weight=weights)
            loss = lossf(outputs, answers)

            out = torch.sigmoid(outputs).cpu().detach().numpy()
            
            #building grid_prot, grid_out
            iidxs = [xs.index(vec_starts[0][0]),ys.index(vec_starts[0][1]),zs.index(vec_starts[0][2])]
            
            g_i = [8+48*iidx     for iidx in iidxs] #grid_all_init(x,y,z)
            g_f = [8+48*(iidx+1) for iidx in iidxs] #grid_all_final(x,y,z)
            w_i = [8            for iidx in iidxs] #wgs_init(x,y,z)
            w_f = [56           for iidx in iidxs] #wgs_final(x,y,z)      
       
            #mod for boundary 
            for iidx_i , iidx in enumerate(iidxs):
                if iidx == 0:
                    g_i[iidx_i] -= 8 
                    w_i[iidx_i] -= 8 
                elif iidx == (len_idxs[iidx_i] -1):
                    g_f[iidx_i] += 8 
                    w_f[iidx_i] += 8 
        
            grid_out[:,:,g_i[0]:g_f[0], g_i[1]:g_f[1], g_i[2]:g_f[2]] = out[:,:,w_i[0]:w_f[0], w_i[1]:w_f[1], w_i[2]:w_f[2]]
            grid_prot[:,:,g_i[0]:g_f[0], g_i[1]:g_f[1], g_i[2]:g_f[2]] = np_inputs[:,:,w_i[0]:w_f[0], w_i[1]:w_f[1], w_i[2]:w_f[2]]

        grid_vs = np.array([[xs[0],ys[0],zs[0]]]) 
        wat_dict = run_place_water_part(grid_vs, grid_prot, grid_out,vcut=2.00,grid=0.5,padding=padding )
        #fpath = '%s/%s.pdb'%(dr,idx)
        fpath = '%s.pdb'%(idx)
        write_out_pdb(fpath,wat_dict['vecs'],wat_dict['scores'])
        time_end = time.time()
        if tlog == True:
            tdiff = time_end - time_start
            time_log.write('%s %15.7f\n'%(idx,tdiff))
    if tlog == True:
        time_log.close()
    
def run_epoch(env, epoch, train=True,build=False, batchsize=4 ,dr=None):
    idxs      = env['idxs']
    net       = env['net']
    device    = env['device']
    optimizer = env['optimizer']
    log       = env['log']
    n_grid    = env['n_grid']
    build_r   = env['build_r']
    dbloss    = env['dbloss']
    running_loss = 0.0
    total_loss = 0.0

    if build:
        batchs = batch_idxs(idxs, batch=1, shuffle= False)
    elif train:
        batchs = batch_idxs(idxs, batch=batchsize, shuffle= True)
    else:
        batchs = batch_idxs(idxs, batch=batchsize, shuffle= False)
    
    for i, batch in enumerate(batchs):
        if build:
            vec_starts, np_inputs,np_answers,np_weights = build_batch_all(batch,padding=build_r,dbloss=dbloss) 
        elif train: 
            vec_starts, np_inputs,np_answers,np_weights = build_batch(batch,train=True,n_grid=n_grid,dbloss=dbloss)
        else:
            vec_starts, np_inputs,np_answers,np_weights = build_batch(batch,train=False,n_grid=n_grid,dbloss=dbloss)
            

        inputs = torch.tensor(torch.FloatTensor(np_inputs), device=device ,requires_grad=False, dtype=torch.float32)
        answers = torch.tensor(torch.FloatTensor(np_answers), device=device ,requires_grad=False, dtype=torch.float32)
        weights = torch.tensor(torch.FloatTensor(np_weights), device=device ,requires_grad=False, dtype=torch.float32)
        inputs = inputs.to(device=device) #gpu
        answers = answers.to(device=device) #gpu
        weights = weights.to(device=device) #gpu
    
        if build or (not train): 
            outputs = net.eval()(inputs) 
            lossf = nn.BCEWithLogitsLoss(weight=weights)
            loss = lossf(outputs, answers)
        else:
            optimizer.zero_grad()
            outputs = net.train()(inputs) 
            lossf = nn.BCEWithLogitsLoss(weight=weights)
            loss = lossf(outputs, answers)
            loss.backward()
            optimizer.step()

        if build:
            out = torch.sigmoid(outputs).cpu().detach().numpy()
            run_place_water(vec_starts, np_inputs, out, batch, dr ,vcut=2.00,grid=0.5 )
    
        # print statistics
        running_loss += loss.item()
        total_loss   += loss.item()
        if train and (not build):
            if i % 5 == 4:
                print('T[%d, %5d] loss: %.6f' %
                      (epoch + 1, i + 1, running_loss/5.))
                log.write('T[%d, %5d] loss: %.6f\n' %
                      (epoch + 1, i + 1, running_loss/5.))
    
                running_loss = 0.0
        print (batch)

    if (not train) and (not build):
       print('E[%d, %5d] loss: %.6f' %
             (epoch + 1, i + 1, total_loss/float(len(batchs))))
       log.write('E[%d, %5d] loss: %.6f\n' %
             (epoch + 1, i + 1, total_loss/float(len(batchs))))

def training(nets,names ,n_grid=32, build_r=32., build_prefixs = [],dbloss=False,batchsize=4):
    for idx, net in enumerate(nets):
        name = names[idx]
        curr_dir =  os.path.dirname(__file__)#new 
        cwd = os.getcwd()  #new
        os.chdir(curr_dir) #new
        if not os.access(name,0):
            os.mkdir(name)
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Prakhar
        device = "cpu"
        #
        net.to(device)
        if torch.cuda.device_count() >1:
            net = nn.DataParallel(net)
        
        states = glob.glob('%s/cifar_state_*.bin'%(name))
        states.sort()
    
        if len(states) > 0:
            start_epoch = int(states[-1].split('/')[-1].split('.')[0].split('_')[-1])
            net.load_state_dict(torch.load(states[-1]))
        else:
            start_epoch = 0
        os.chdir(cwd) #new        
        optimizer = optim.Adam(net.parameters(), lr=0.0001)
        
        log = open('%s/cnn_net_v%d.log'%(name,idx),'a')
        env_train = {'idxs':trainidxs, 'net':net, 'device':device,'dbloss':dbloss,
                     'optimizer':optimizer, 'log':log,'n_grid':n_grid,'build_r':build_r}
        
        env_test = {'idxs':testidxs, 'net':net, 'device':device,'dbloss':dbloss,
                     'optimizer':optimizer, 'log':log,'n_grid':n_grid,'build_r':build_r}
        
        for epoch in range(start_epoch, 1000):  # loop over the dataset multiple times
            train_epoch(env_train, epoch,batchsize=batchsize)
            test_epoch(env_test, epoch)
                 
            if epoch%5 ==4:
                test_epoch(env_test, epoch)

            if epoch%50 == 49:
                #save state:
                torch.save(net.module.state_dict(), '%s/cifar_state_%05d.bin'%(name,(epoch+1)))
    
        log.close()
        del(log)
        print('BUILD_TRAIN')
        build_epoch(env_train, 1000, train=True, prefix=build_prefixs[idx])
        print('BUILD_TEST')
        build_epoch(env_test , 1000, train=False, prefix=build_prefixs[idx])

def predict_path(nets,names,in_path,out_name,n_grid=64, padding=4.0, build_prefixs = [],dbloss=False,tlog=False):
    for idx, net in enumerate(nets):
        name = names[idx]
        curr_dir =  os.path.dirname(__file__)#new 
        cwd = os.getcwd()  #new
        os.chdir(curr_dir) #new
        if not os.access(name,0):
            os.mkdir(name)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        if torch.cuda.device_count() >1:
            net = nn.DataParallel(net)
        
        states = glob.glob('%s/cifar_state_*.bin'%(name))
        states.sort()
    
        if len(states) > 0:
            start_epoch = int(states[-1].split('/')[-1].split('.')[0].split('_')[-1])
            net.load_state_dict(torch.load(states[-1]))
        else:
            start_epoch = 0
        os.chdir(cwd) #new        
        optimizer = optim.Adam(net.parameters(), lr=0.0001)
        pd = {}
        pd[out_name] = {'pro':[in_path] , 
                         'wat':[]} 
        env = {'idxs':[out_name], 'net':net, 'device':device,'dbloss':dbloss,
                     'optimizer':optimizer, 'n_grid':n_grid,'padding':padding,
                     'use_paths':True, 'paths_dict':pd}
        build_full_epoch(env, 1000, train=False, prefix=build_prefixs[idx],tlog=tlog)

def predict_path_cpu(nets,names,in_path,out_name,n_grid=64, padding=4.0, build_prefixs = [],dbloss=False,tlog=False,tlog_path='cpu_time.log'):
    for idx, net in enumerate(nets):
        name = names[idx]
        curr_dir =  os.path.dirname(__file__)#new 
        cwd = os.getcwd()  #new
        os.chdir(curr_dir) #new
        if not os.access(name,0):
            os.mkdir(name)
        
        states = glob.glob('%s/cifar_state_*.bin'%(name))
        states.sort()
    
        if len(states) > 0:
            start_epoch = int(states[-1].split('/')[-1].split('.')[0].split('_')[-1])
            net.load_state_dict(torch.load(states[-1],map_location='cpu'))
        else:
            start_epoch = 0
        os.chdir(cwd) #new        
        optimizer = optim.Adam(net.parameters(), lr=0.0001)
        pd = {}
        pd[out_name] = {'pro':[in_path] , 
                         'wat':[]} 
        env = {'idxs':[out_name], 'net':net, 'device':'cpu','dbloss':dbloss,
                     'optimizer':optimizer, 'n_grid':n_grid,'padding':padding,
                     'use_paths':True, 'paths_dict':pd}
        build_full_epoch(env, 1000, train=False, prefix=build_prefixs[idx],tlog=tlog,tlog_path=tlog_path)

# if __name__ == '__main__':
#     #idxs - For training (requires GPU)
    
#     #idxs is list of idx, where idx is the file name of PDB, used like below
#     #   paths = {'pro':['./pdb/pdb_protein/%s.pdb'%(idx)],
#     #           'wat':['./pdb/pdb_water/%s.pdb'%(idx)]}
#     # pdb_protein contains protein only PDB files.
#     # pdb_water   contains water only PDB files - for training
    
#     trainidxs  = []
#     testidxs   = []
    
#     if len(sys.argv) != 3:
#         print('usage: GWCNN_cpu.py [input PDB/mmCIF] [output name]')
        
#     else:    
#         pro_path = sys.argv[1]
#         out_name = sys.argv[2]
#         tlog_path ='cpu_time.log'
#         nets  = [networks.Net_v4_5_auxloss()]
#         names = ['networks'] 
#         predict_path_cpu(nets,names,pro_path,out_name,n_grid=64, padding=4.0, build_prefixs=names,dbloss=True,tlog = False)

if __name__ == '__main__':
    #idxs - For training (requires GPU)
    
    #idxs is list of idx, where idx is the file name of PDB, used like below
    #   paths = {'pro':['./pdb/pdb_protein/%s.pdb'%(idx)],
    #           'wat':['./pdb/pdb_water/%s.pdb'%(idx)]}
    # pdb_protein contains protein only PDB files.
    # pdb_water   contains water only PDB files - for training
    print("current dir: ", os.getcwd())  
    trainidxs = ['1bhe','1bkr','1chd','1dus','1dzf','1es5','1ew4','1f32','1fcq','1gqe','1iib','1jhs','1jyh','1k4n','1lfp','1m1s','1mix','1mk4','1msc','1n7k','1nar','1psw','1qw2','1ri6','1sgm','1sh8','1t6t','1tp6','1tua','1tvx','1uek','1v6z','1v77','1vin','1wer','1wwi','1x2i','1xaw','1xfs','1xkr','1y9w','1ylx','1yn3','1ypy','1zvb','2axo','2c2i','2cwq','2cxd','2d59','2d5d','2dyi','2e8g','2ebe','2erf','2f23','2fhz','2fq3','2gau','2gs5','2gxg','2hxi','2hy5','2ig8','2nnu','2nr7','2pv2','2qgm','2rbb','2vga','2x4j','2yvt','2yxm','2zq5','2zvy','3aq2','3bcy','3bpv','3bqx','3bs7','3c5v','3d79','3dgp','3enu','3fb9','3fcd','3ffv','3g1j','3g8k','3ggy','3grh','3h2g','3h6r','3ho7','3hvw','3k8u','3m5b','3nph','3q1c','3q64','3qf2','3rvc','3vzh','3wvt','3zbd','4djg','4evf','4g29','4gf3','4gvb','4ikn','4j4r','4jkz','4jqf','4kia','4mis','4pbo','4pqd','4rth','4xe7','4ywz','4zpc','5b0u','5b6c','5c5z','5dae','5fce','5h28','5i5n','5i8j','5itm','5j4f','5jge','5l37','5lnd','5u96','5vtl','5wd8','5xbc','5y6h','5yqi','5z6d','6a5c','6a5h','6ba9','6cca','6cva','6e4o','6e7e','6fm5','6i50','6jny','6k1w','6p28','6pym','6rzy']

    testidxs = ['1a62','1a8q','1arb','1atz','1b8p','1cq3','1dvo','1eg3','1es9','1f00','1f46','1g61','1gak','1h99','1hzt','1i60','1j7g','1jb3','1jl1','1jy2','1jyk','1kve','1kxo','1n1j','1ng6','1o8x','1o9g','1ow1','1q5z','1s4k','1sdo','1se8','1u7b','1ufi','1usg','1v6t','1vaj','1weh','1whi','1wna','1y12','1yac','1ylm','1z4e','2bjq','2cvb','2cxh','2d4p','2db7','2dp9','2dtc','2dyj','2e12','2ej8','2end','2f5g','2f6e','2fl4','2fzp','2hpl','2hrz','2iu1','2odl','2ooa','2p4h','2p58','2pge','2pof','2qjz','2qk1','2r2c','2rb8','2x4l','2xhf','2xpp','2yv4','2yva','2yvs','2yyv','2z14','2zca','2zcm','3a4c','3c90','3cnu','3d1b','3dfg','3dz1','3e0h','3efy','3eoi','3etv','3fau','3fke','3hqx','3hut','3hxl','3ils','3kgk','3kp8','3kvd','3l3f','3lw3','3m66','3m8j','3nj2','3nrw','3obq','3p9v','3pid','3rv1','3s8m','3u4v','3vhj','3vz9','4b9g','4cbe','4d53','4dzo','4g9q','4giw','4gs3','4ic4','4j0w','4me2','4rs7','4uds','4xy5','4z2n','4zbh','5dtc','5e71','5fid','5hyz','5mc7','5njo','5u2l','5x9k','5xaq','5ygh','5zt3','6aht','6ar0','6bus','6bw9','6d0a','6dnm','6h8f','6ivc','6j56','6k93','6l1m']
    
    nets = [networks.Net_v4_5_auxloss()]
    
    # nets  = []
    names = ['networks1']
    training(nets,names, dbloss=True)
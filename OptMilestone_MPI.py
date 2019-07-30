from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

import numpy as np
import pandas as pd
import sys

## correct (Xp, Yp) and (Xn, Yn)
def CorrectXYnp(Xp,Yp,Xn,Yn):
    for i in range(1,len(slope_opt)):
        d1 = (Xp[i] - Xp[i-1])**2 + (Yp[i] - Yp[i-1])**2
        d2 = (Xn[i] - Xp[i-1])**2 + (Yn[i] - Yp[i-1])**2
        d3 = (Xn[i] - Xn[i-1])**2 + (Yn[i] - Yn[i-1])**2
        d4 = (Xp[i] - Xn[i-1])**2 + (Yp[i] - Yn[i-1])**2
        if max(d2,d4) <  max(d1,d3):
            tmpx, tmpy = [Xp[i].copy(), Yp[i].copy()]
            tmpx1, tmpy1 = [Xn[i].copy(), Yn[i].copy()]
            Xp[i], Yp[i], Xn[i], Yn[i] = [tmpx1, tmpy1, tmpx, tmpy]

# rotation matrix from each milestone to x-axis
def RotationMatrix(Xp,Yp,X,Y,r):
    RotM = []
    vec0 = np.array([1,0])
    for i in range(len(X)):
        vec1 = np.array([Xp[i]-X[i], Yp[i]-Y[i]])
        theta = np.arccos( np.dot(vec1.T, vec0)/r )

        #if vec1[0]<0 and vec1[1]<0:
        #idx = int(np.minimum(i+1,len(X)-1))
        #if (X[idx] < X[i]) and (vec1[1]<0):
        #    theta = 2*np.pi-theta
        if vec1[1]<0:
            theta = 2*np.pi-theta

        c, s = [np.cos(theta), np.sin(theta)]
        rot = np.array([[c, s],[-s, c]]).reshape(2,2)
        RotM.append(rot)
    RotM = np.asarray(RotM)

    return(RotM)

##############################

if rank == 0:
    print("Reading input files from the disk ...")
    dir='/hdd/si_han/Project/CDK8/Milestoning_metadynamics/CDK8_CycC_PL2/Diff_Milestones0/'
    dir2='/hdd/si_han/Project/CDK8/Milestoning_metadynamics/CDK8_CycC_PL2/Projections/'
    path = pd.read_csv(dir+'finalpath.txt', header=None, delimiter=r"\s+")
    #path = pd.read_csv('/hdd/si_han/GPU0/sh_tzy/CDK8/milestoning/metadynamics/CDK8_Cyclin_Ligand1/milestone_resave/finalpath.txt', header=None, delimiter=r"\s+")
    fp = np.asarray(path)
    fp = fp[:,1:3]
    x = fp[:,0].reshape(len(fp),1)
    y = fp[:,1].reshape(len(fp),1)

    slope_opt = np.load(dir+'milestone_slopes.npy')
    #slope_opt = pd.read_csv('/hdd/si_han/Tools/MILESTONE_ONLY/Try/3.a/milestone2.txt',header=None,delimiter=r"\s+")
    #slope_opt = np.asarray( np.tan(slope_opt.iloc[:,1]/180*np.pi) ).reshape(len(fp),1)
    PCA1 = np.load(dir2+'PCA1_300_0.npy')
    PCA2 = np.load(dir2+'PCA2_300_0.npy')
    IDX = np.load(dir2+'IDX_300_0.npy')
    MIDX_ALL = []

    # slope matrix NxN
    m = np.zeros((len(slope_opt),len(slope_opt)))
    np.fill_diagonal(m,slope_opt)

    # boundary of milestones in x-axis Nx1
    #sc=15
    sc=float(sys.argv[2])
    dx = np.sqrt((sc/2)**2/(1+slope_opt**2))
    xp = x + dx
    xn = x - dx

    # boundary of milestones in y-axis Nx1
    yp = y + np.dot(m,dx)
    yn = y - np.dot(m,dx)

    CorrectXYnp(xp,yp,xn,yn)
    if (x[1]-x[0] > 0 and yp[0]-y[0]<0) or (x[1]-x[0] < 0 and yp[0]-y[0]>0):
        A, B = [xp.copy(), yp.copy()]
        C, D = [xn.copy(), yn.copy()]
        xp, yp, xn, yn = [C, D, A, B]
    
    rotM=RotationMatrix(xp,yp,x,y,sc/2)

    NFRAMES = len(PCA1)

else:
    x, y, NFRAMES, rotM, slope_opt = [None]*5
    
NFRAMES = comm.bcast(NFRAMES, root=0)
CHUNKSIZE = 100000
N_CHUNKS = int(np.ceil(NFRAMES/CHUNKSIZE))
#N_CHUNKS = 3
x = comm.bcast(x, root=0)
y = comm.bcast(y, root=0)
rotM = comm.bcast(rotM, root=0)
slope_opt = comm.bcast(slope_opt, root=0)


for n in range(N_CHUNKS):
    if rank == 0:
        print("\nCHUNK %d / %d" %(n+1,N_CHUNKS))
        ISTART = n*CHUNKSIZE
        IEND = int( min( (n+1)*CHUNKSIZE, NFRAMES ) )
        pca1 = PCA1[ISTART:IEND]
        pca2 = PCA2[ISTART:IEND]
        idx = IDX[ISTART:IEND]
        nframes = len(pca1)
        midx = np.full(len(pca1),2*len(x))
    else:
        nframes = None

    nframes = comm.bcast(nframes, root=0)
    chunksize = int(np.ceil(nframes/size))

    if rank==0:
        print("\tsending pca1 ... ")
    for i in range(1, size):
        istart = int( i*chunksize )
        iend = int( min((i+1)*chunksize, nframes) )
        if rank == 0:
            pca1_ = pca1[istart:iend]
            comm.Isend(pca1_, dest=i, tag=4)
        if rank == i:
            pca1_ = np.arange( iend-istart  , dtype=np.float64)
            comm.Recv(pca1_, source=0, tag=4)

    if rank==0:
        print("\tsending pca2 ... ")         
    for i in range(1, size):
        istart = int( i*chunksize )
        iend = int( min((i+1)*chunksize, nframes) )
        if rank == 0:
            pca2_ = pca2[istart:iend]
            comm.Isend(pca2_, dest=i, tag=5)
        if rank == i:
            pca2_ = np.arange( iend-istart  , dtype=np.float64)
            comm.Recv(pca2_, source=0, tag=5)       

    if rank==0:
        print("\tsending idx ... ")
    for i in range(1, size):
        istart = int( i*chunksize )
        iend = int( min((i+1)*chunksize, nframes) )
        if rank == 0:
            idx_ = idx[istart:iend]
            comm.Isend(idx_, dest=i, tag=6)
        if rank == i:
            idx_ = np.arange( iend-istart  , dtype=np.int)
            comm.Recv(idx_, source=0, tag=6) 

    if rank==0:
        print("\tsending midx ... ")
    for i in range(1, size):
        istart = int( i*chunksize )
        iend = int( min((i+1)*chunksize, nframes) )
        if rank == 0:
            midx_ = midx[istart:iend]
            comm.Isend(midx_, dest=i, tag=7)
        if rank == i:
            midx_ = np.arange( iend-istart  , dtype=np.int)
            comm.Recv(midx_, source=0, tag=7)

    if rank==0:
        istart = int( rank*chunksize )
        iend = int( min((rank+1)*chunksize, nframes) )
        pca1_ = pca1[istart:iend]
        pca2_ = pca2[istart:iend]
        idx_ = idx[istart:iend]
        midx_ = midx[istart:iend]

#########################################

    if rank == 0:
        print("\tInitializing calculation in %d MPI threads" %size) 
        print("\tNumber of data in each thread: %d" % len(pca1_))

    radius = float(sys.argv[2])/2
    rcut = radius**2

    for i in range(len(pca1_)):
        D = (pca1_[i]-x)**2+(pca2_[i]-y)**2
        index = np.argmin(D)
        imin = max(0, index-20)
        imax = min(len(slope_opt)-1, index+20)

        # Add more points between milestones to avoid gaps
        D2 = (pca1_[i] - (x[:-1]+x[1:])/2)**2 + (pca2_[i] - (y[:-1]+y[1:])/2)**2
        #if (min(D)>rcut):
        if min(D)>rcut and min(D2)>rcut:
            continue
        else:
            for j in range(imax-imin+1):
                k = imin+j
                vec = np.array([pca1_[i]-x[k], pca2_[i]-y[k]])
                vec_r = np.dot(rotM[k], vec)

                if vec_r[1]>0:
                    if j == 0:
                        midx_[i]=0
                    else:
                        midx_[i]=k
                    break
                elif k == len(slope_opt)-1:
                    midx_[i]=len(slope_opt)
        if rank == 0:
            if i%1000 == 0:
                print("\tprogress: %d / %d" %(i, len(pca1_)), end="\r")
                True
            elif i == len(pca1_)-1:
                print("\tprogress: %d / %d" %(i+1, len(pca1_)))
                print("\tSending results to the root ... \n")

    MIDX1 = comm.gather(midx_, root=0)
    # gathering results
    if rank == 0:
        MIDX1 = np.concatenate(MIDX1,axis=0)
        MIDX_ALL.append(MIDX1)

    comm.Barrier()

if rank == 0:
    print("Combing results from all the CHUNKS\n")
    MIDX_ALL = np.concatenate(MIDX_ALL,axis=0)
    print(MIDX_ALL.shape)
    filename = sys.argv[1]
    np.save(filename, MIDX_ALL)


#filename='/hdd/si_han/Tools/MILESTONE_ONLY/Try/3.a/MIDX.npy'

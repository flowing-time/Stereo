# %%
import numpy as np
import cv2
import maxflow
import os

INPUT_DIR = 'images'
OUTPUT_DIR = './output'

# %%
def disp_cost(im, im_disp, p):
    # im.dtype and im_disp.dtype are float

    shift = np.zeros_like(im_disp)

    if p == 0:
        shift = im_disp.copy()
    elif p > 0:
        shift[:, p:] = im_disp[:, :-p]
    else:
        shift[:, :p] = im_disp[:, -p:]

    cost = np.sum((im - shift) ** 2, axis=2)

    return cost


# %%
def disp_energy(im, im_disp, ndisp=0):

    if ndisp == 0:
        ndisp = im_disp.shape[1] // 4

    im, im_disp = im.astype('float'), im_disp.astype('float')

    D = np.zeros((*im.shape[:2], abs(ndisp)), dtype='float')
    for p in range(abs(ndisp)):
        D[:, :, p] = disp_cost(im, im_disp, p if ndisp > 0 else -p)

    return D
    
    
# %%
# Sum of Squared Differences disparity map
def get_ssd_dmap(D, w=5):

    ssd = np.zeros_like(D)
    for p in range(D.shape[-1]):
        ssd[:, :, p] = cv2.boxFilter(D[:, :, p], -1, (w, w),
                    normalize=False, borderType=cv2.BORDER_ISOLATED)

    dmap = ssd.argmin(axis=-1)

    return dmap

# %%
def V_Potts(D, k=50.0):

    L = D.shape[-1]

    return k * (np.ones(L) - np.eye(L))

# %%
def V_L2norm(D, k=50.0, beta=1.0):

    L = D.shape[-1]
    lx, ly = np.meshgrid(range(L), range(L))
    V = beta * abs(lx - ly).astype('float')
    V[V > k] = k

    return V

# %%
def alpha_expansion(alpha, D, V, labels):

    rows, cols = labels.shape

    label_weights = V[alpha, labels]

    g = maxflow.Graph[float]()

    # Build graph below: add pixel nodes, auxiliary nodes, t-links and n-links.

    # Add pixel nodes
    node_ids = g.add_grid_nodes(labels.shape)

    # T-links for pixel nodes
    T_alpha = D[:, :, alpha]

    #T_n_alpha = np.ones_like(T_alpha)
    #for i in range(rows):
    #    for j in range(cols):
    #        T_n_alpha[i, j] = D[i, j, labels[i, j]]
    T_n_alpha = D[(*np.mgrid[:rows, :cols], labels)]
    T_n_alpha[labels == alpha] = np.inf

    g.add_grid_tedges(node_ids, T_alpha, T_n_alpha)

    # Add horizontal auxiliary nodes
    ha_ids = g.add_grid_nodes((rows, cols-1))

    # T-links for horizontal auxiliary nodes
    T_ha_n_alpha = V[labels[:, :-1], labels[:, 1:]]
    g.add_grid_tedges(ha_ids, 0, T_ha_n_alpha)

    # Horizontal n-links
    node_ha_ids = np.empty((rows, 2 * cols - 1), dtype='int')
    node_ha_ids[:, ::2] = node_ids
    node_ha_ids[:, 1::2] = ha_ids

    h_weights = np.zeros_like(node_ha_ids, dtype='float')
    h_weights[:, ::2] = label_weights

    g.add_grid_edges(node_ha_ids, h_weights, np.array([0, 0, 1]), symmetric=True)
    g.add_grid_edges(node_ha_ids, h_weights, np.array([1, 0, 0]), symmetric=True)

    # Add vertical auxiliary nodes
    va_ids = g.add_grid_nodes((rows-1, cols))

    # T-links for vertical auxiliary nodes
    T_va_n_alpha = V[labels[:-1, :], labels[1:, :]]
    g.add_grid_tedges(va_ids, 0, T_va_n_alpha)

    #  Vertical n-links
    node_va_ids = np.empty((2 * rows - 1, cols), dtype='int')
    node_va_ids[::2, :] = node_ids
    node_va_ids[1::2, :] = va_ids

    v_weights = np.zeros_like(node_va_ids, dtype='float')
    v_weights[::2, :] = label_weights

    g.add_grid_edges(node_va_ids, v_weights, np.array([[0], [0], [1]]), symmetric=True)
    g.add_grid_edges(node_va_ids, v_weights, np.array([[1], [0], [0]]), symmetric=True)

    # Run graph cut to get min energy and update labels
    energy = g.maxflow()
    labels[g.get_grid_segments(node_ids)] = alpha

    return energy

# %%
# Graph cut alpha expansion disparity map
def get_ae_dmap(D, V, labels=None, cycle_max=5):
    # cycle_max is default to 5.
    # Usually after 5 cycles, further energy reduction is very small.
    # If cycle_max is set to 0, there is no upper limit of cycle times.
    if cycle_max is 0:
        cycle_max = np.inf

    dmap = D.argmin(axis=-1) if labels is None else labels.copy()

    ndisp = D.shape[-1]
    Ef = np.inf
    cycle = 0
    print('INFO: Started alpha expansion cycle. It may take several minutes...')
    while True:

        success = False
        for alpha in range(ndisp):
            Ef_hat = alpha_expansion(alpha, D, V, dmap)
            if Ef_hat < Ef:
                Ef = Ef_hat
                success = True
            
        cycle += 1
        print('Debug:\tcycle %d\tEf %.1f' % (cycle, Ef))

        if not success or cycle >= cycle_max:
            break

    print('INFO: Finished alpha expansion with %d cycles.' % cycle)

    return dmap

# %%
def save_dmap(dmap, id_string, id_num):

    norm_dmap = cv2.normalize(dmap, None, 0, 255, cv2.NORM_MINMAX)

    filename = id_string + str(id_num) + '.png'

    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), norm_dmap)
    

# %%
def main():

    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # SSD algorithm results
    img0 = cv2.imread('images/cones/im2.png')
    img1 = cv2.imread('images/cones/im6.png')

    DL = disp_energy(img0, img1, 80)
    DR = disp_energy(img1, img0, -80)

    dmap5L = get_ssd_dmap(DL)
    dmap5R = get_ssd_dmap(DR)
    dmap19L = get_ssd_dmap(DL, 19)
    dmap19R = get_ssd_dmap(DR, 19)

    save_dmap(dmap5L, 'Cone_ssd_L_dmap_w', 5)
    save_dmap(dmap5R, 'Cone_ssd_R_dmap_w', 5)
    save_dmap(dmap19L, 'Cone_ssd_L_dmap_w', 19)
    save_dmap(dmap19R, 'Cone_ssd_R_dmap_w', 19)


    # Alpha expansion algorithm results
    scenes = ['Adirondack', 'Jadeplant', 'Motorcycle',
                'Piano', 'Pipes', 'Playroom', 'Playtable',
                'Recycle', 'Teddy', 'Vintage', 'Shelves']

    ndisps = [73, 160, 70,
                65, 75, 83, 73,
                65, 64, 190, 60]

    for scene, ndisp in zip(scenes, ndisps):
        print('\n\nINFO: Working on scene %s with maximum disparity value %d...' % (scene, ndisp))

        img0 = cv2.imread(os.path.join(INPUT_DIR, scene, 'im0.png'))
        img1 = cv2.imread(os.path.join(INPUT_DIR, scene, 'im1.png'))

        D = disp_energy(img0, img1, ndisp)

        # Get ssd dmap with two different window sizes: 5 and 19
        w_sizes = [5, 19]
        for w in w_sizes:
            ssd_dmap = get_ssd_dmap(D, w)
            save_dmap(ssd_dmap, scene + '_ssd_dmap_w', w)

        # Get alpha expansion dmap with different smoothness terms V
        Vs = [V_Potts(D, 200), V_Potts(D, 500), V_L2norm(D, k=600, beta=40)]

        for i, V in enumerate(Vs):
            print('\nINFO: Smoothness term V%d:' % i)

            ae_dmap = get_ae_dmap(D, V, ssd_dmap, cycle_max=5)
            save_dmap(ae_dmap, scene + '_ae_dmap_V', i)
        


# %%
if __name__ == "__main__":
    main()

import numpy as np

def interpolate(x1, x2, in_num):
    warp_pose = []
    warp_pose.append(x1)
    if in_num == 1:
        interp_pose = (x1+x2)/2
        warp_pose.append(interp_pose)
    else:
        avg_num = in_num + 1
        for i in range(1, avg_num):
            warp_pose.append(x2*i/(avg_num) + x1*(avg_num-i)/avg_num)
    # warp_pose.append(x2)
    return warp_pose

def interp(x, w):
    # x: raw skeleton T, 22, 3, w: warpping frames list[t1, t2, t3]
    new_skel = [x[0]]
    total_n = len(w)
    t_start = 0
    while t_start < total_n - 1:
        start_val = w[t_start]
        t_end = t_start + 1
        while t_end < total_n - 1 and w[t_end] == start_val:
            t_end += 1
        if t_end - t_start == 1:
            old_frame_n = w[t_start]
            new_skel.append(x[old_frame_n])
            t_start += 1
        else:
            interval = t_end - t_start
            # print(t_start, t_end, interval)
            old_frame_1 = x[w[t_start]]
            old_frame_2 = x[w[t_end]]
            warp_poses = interpolate(old_frame_1, old_frame_2, interval - 1)
            # print(len(warp_poses))
            # print('------------')
            new_skel.extend(warp_poses)
            t_start = t_end
    new_skel.append(x[w[t_start]])
    return np.stack(new_skel)
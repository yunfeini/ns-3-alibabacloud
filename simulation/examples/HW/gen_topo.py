with open('/home/kumo/workspace/ns-3-alibabacloud/' \
'simulation/examples/HW/topo.txt', 'w') as f:
    M = 4
    N = 16
    B = '200Gbps'
    G = 128//N

    NODES = 64*N*G
    SS = 8192*M
    EPS = 4*M*G
    XPU_GRP = 64*N

    L_O = '200ns'
    L_L0 = '200ns'
    L_L1 = '600ns'
    L_L2 = '1000ns'
    LINKS = 122752

    f.write(f'{NODES+SS+EPS} 1 0 {SS+EPS} {LINKS} X000 8\n')
    for i in range(SS+EPS):
        f.write(f'{i+NODES} ')
    #L0
    for g in range(G):
        for n in range(N):
            for tray in range(8):
                for i in range(8):
                    index = g*XPU_GRP + n*64 + tray*8
                    for j in range(8):
                        if i != j:
                            f.write(f'{index+i} {index+j} {B} {L_O} 0\n')

    for g in range(G):
        for n in range(N):
            for m in range(M):
                for tray in range(8):
                    for i in range(8):
                        index = g*XPU_GRP + n*64 + tray*8 + i
                        idx_eps = NODES + g*M*N*8 + n*8*M + 8*m + tray
                        f.write(f'{index} {idx_eps} {B} {L_L0} 0\n')
    #L1
    for g in range(G):
        for n in range(N):
            for m in range(M):
                for i in range(8):
                    for j in range(4):
                        index = NODES + g*M*N*8 + n*M*8 + m*8 + i
                        idx_eps = NODES + SS + g*M*4 + m*4 + j
                        f.write(f'{index} {idx_eps} {B} {L_L1} 0\n')
    #L2
    for i in range(EPS):
        for j in range(EPS):
            if i != j:
                index = NODES + SS + i
                idx_eps = NODES + SS + j
                f.write(f'{index} {idx_eps} {B} {L_L2} 0\n')
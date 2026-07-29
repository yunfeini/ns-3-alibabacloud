with open('/home/kumo/workspace/ns-3-alibabacloud/' \
'simulation/examples/HW/topo1_8_4.txt', 'w') as f:
# 'simulation/examples/HW/topo1_16_2.txt', 'w') as f:
    B = '400Gbps'
    M = 8

    N = 1

    # G = 16
    # T = 2

    G = 8
    T = 4

    NODES = 8*T*N*G
    SS = T*M*N*G
    EPS = 4*M*G
    XPU_GRP = 8*T*N

    L_O = '0ns'
    L_L0 = '0ns'
    L_L1 = '850ns'
    L_L2 = '1205ns'
    B_OCS = '1600Gbps'

    LINKS = -1
    LINKS = 71167

    f.write(f'{NODES+SS+EPS} 1 0 {SS+EPS} {LINKS} X000 8\n')
    for i in range(SS+EPS):
        f.write(f'{i+NODES} ')
    #L0
    for g in range(G):
        for n in range(N):
            for tray in range(T):
                for i in range(8):
                    index = g*XPU_GRP + n*8*T + tray*8
                    for j in range(8):
                        if i != j:
                            f.write(f'{index+i} {index+j} {B} {L_O} 0\n')
                            LINKS += 1

    for g in range(G):
        for n in range(N):
            for m in range(M):
                for tray in range(T):
                    for i in range(8):
                        index = g*XPU_GRP + n*8*T + tray*8 + i
                        idx_eps = NODES + g*M*N*T + n*T*M + m*T + tray
                        f.write(f'{index} {idx_eps} {B} {L_L0} 0\n')
                        LINKS += 1
                        
    #L1
    for g in range(G):
        for n in range(N):
            for m in range(M):
                for i in range(8):
                    for j in range(4):
                        index = NODES + g*M*N*T + n*T*M + m*T + i
                        idx_eps = NODES + SS + g*M*4 + m*4 + j
                        f.write(f'{index} {idx_eps} {B} {L_L1} 0\n')
                        LINKS += 1
                        
    #L2
    for i in range(EPS):
        for j in range(EPS):
            if i != j:
                index = NODES + SS + i
                idx_eps = NODES + SS + j
                f.write(f'{index} {idx_eps} {B_OCS} {L_L2} 0\n')
                LINKS += 1

    print(LINKS)
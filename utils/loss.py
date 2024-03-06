import torch


def CCC_loss(x, y):
    y = y.view(-1)
    x = x.view(-1)
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    rho = torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2)))+1e-8)
    x_m = torch.mean(x)
    y_m = torch.mean(y)
    x_s = torch.std(x)
    y_s = torch.std(y)
    ccc = 2*rho*x_s*y_s/(torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2))
    return 1-ccc, ccc


def VA_loss(vout, aout, label):
    vout = torch.clamp(vout, -1, 1)     # [bs, sq, 1]
    aout = torch.clamp(aout, -1, 1)     # [bs, sq, 1]
    bz, seq, _ = vout.shape
    label = label.view(bz * seq, -1)        # [bs, sq, 2]
    vout = vout.view(bz * seq, -1)
    aout = aout.view(bz * seq, -1)

    ccc_valence_loss, ccc_valence = CCC_loss(vout[:, 0], label[:, 0])
    ccc_arousal_loss, ccc_arousal = CCC_loss(aout[:, 0], label[:, 1])

    # logging.info(f"ccc_valence_loss:{ccc_valence_loss:.4}, ccc_arousal_loss:{ccc_arousal_loss:.4}")

    ccc_loss = ccc_valence_loss + ccc_arousal_loss
    ccc_avg = 0.5 * (ccc_valence + ccc_arousal)

    loss = ccc_loss
    return loss, ccc_loss, ccc_avg, [ccc_valence, ccc_arousal]


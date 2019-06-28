from tensorboardX import SummaryWriter
import numpy as np

writer = SummaryWriter()
r = 5
for i in range(10):
    writer.add_scalar('run_14h', i, i)
writer.close()
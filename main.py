import torch
import torch.optim as optim
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from model.Encoder import Encoder
from model.Decoder import Decoder
import os.path as osp



def get_arg():
    parser = argparse.ArgumentParser(description="AutoEncoder")
    parser.add_argument("--data-dir", type=str,
                            default="../Warehouse/atari_data/training", help="path to dataset")
    parser.add_argument("--checkpoint-dir", type=str,
                            default="./checkpoints/", help="path to store checkpoint")
    parser.add_argument("--restore", type=bool,
                            default=False, help="load pretrained model or not")
    parser.add_argument("--batch-size", type=int,
                            default=1000, help="training batch size")
    parser.add_argument("--model-dir", type=str,
                            default="./checkpoints/", help="path in which pretrained weights are stored")
    parser.add_argument("--training", type=bool,
                            default=True, help="to perform training(true) or testing(false)")
    parser.add_argument("--experiment_id", type=int,
                            default=0, help="identification number of this experiment")
    parser.add_argument("--gpu", type=int,
                            default=0, help="choose gpu device.")
    parser.add_argument("--train-size", type=int,
                            default=20000, help="size of training data per load")
    parser.add_argument("--validatin-size", type=int,
                            default=10000, help="size of validation data per load")
    parser.add_argument("--epoch", type=int,
                            default=5, help="number of epoch")
    parser.add_argument("--learning-rate", type=float,
                            default=0.001, help="learning rate of the optimizer")
    return parser.parse_args()
    


def get_data(index):
    train_img = [] # (4-d tensor) shape : size, w, h, 3
    valid_img = []
    batch_img = [] #(4-d tensor) shape: 4, w, h
    directory = args.data_dir
    train_st, train_ed = index, index + args.train_size
    valid_st, valid_ed = train_ed + 1, train_ed + args.validation_size
    for i in range(train_st, train_ed+1):
        if i % 1000 == 0:
            print("loading {:d}-th img".format(i))
        path = directory + '/{:06d}.png'.format(i)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        batch_img.append(img)
        if len(bathc_img) == 4:
            train_img.append(train_img.np.array(output_img, dtype=np.float32))
            batch_img = []
    train_img = np.array(train_img, dtype=np.float32)
    print(train_img.shape)
    for i in range(valid_st, valid_ed+1):
        if i % 1000 == 0:
            print("loading {:d}-th img".format(i))
        path = directory + '/{:06d}.png'.format(i)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        batch_img.append(img)
        if len(batch_img) == 0:
            valid_img.append(valid_img.np.array(valid_img, dtype=np.float32))
            batch_img = []
    valid_img = np.array(valid_img, dtype=np.float32)
    print(valid_img.shape)  
    return train_img, valid_img

#parse command line arguments
args = get_arg()

def main():
    
    #create tensorboard summary writer
    writer = SummaryWriter(args.experiment_id)
    #[TODO] may need to resize input image
    cudnn.enabled = True
    #create model: Encoder
    model_encoder = Encoder()
    model_encoder.train()
    model_encoder.cuda(args.cuda)
    optimizer_encoder = optim.Adam(model_encoder.parameters(), lr=args.learning_rate, betas=(0.95, 0.99))
    optimizer_encoder.zero_grad()

    #create model: Decoder
    model_decoder = Decoder()
    model_decoder.train()
    model_decoder.cuda(args.cuda)
    optimizer_decoder = optim.Adam(model_decoder.parameters(), lr=args.learning_rate, betas=(0.95, 0.99))
    optimizer_decoder.zero_grad()
    
    l2loss = nn.MSELoss()
    
    #load data
    for i in range(1, 210002, 30000):
        train_data, valid_data = get_data(i)
        for e in range(1, args.epoch + 1):
            train_loss_value = 0
            validation_loss_value = 0
            for j in range(0, args.train_size, args.batch_size):
                optimizer_decoder.zero_grad()
                optimizer_decoder.zero_grad()
                latent = model_encoder(train_data[j: j + args.batch_size, :, :, :])
                img_recon = model_decoder(lantent)
                loss = l2loss(img_recon, train_data[j: j + args.batch_size, :, :, :])
                train_loss_value += loss.data.cpu().numpy() / args.batch_size
                loss.backward()
                optimizer_decoder.step()
                optimizer_encoder.step()
            print("data load: {:8d}".format(i))
            print("epoch: {:8d}".format(e))
            print("train_loss: {:08.6f}".format(train_loss_value / (args.train_size / args.batch_size)))
            for j in range(0,args.validation_size, args.batch_size):
                model_encoder.eval()
                model_decoder.eval()
                latent = model_encoder(valid_data[j: j + args.batch_size, :, :, :])
                img_recon = model_decoder(lantent)
                loss = l2loss(img_recon, valid_data[j: j + args.batch_size, :, :, :])
                validatin_loss_value += loss.data.cpu().numpy() / args.batch_size
            model_encoder.train()
            model_decoder.train()
            print("train_loss: {:08.6f}".format(validation_loss_value / (args.validation_size / args.batch_size)))
        torch.save({'deeplab_state_dict': model_deeplab.state_dict()}, osp.join(args.checkpoint_dir, 'AE.pth'))

if __name__ == "__main__":
    main()
                

                


        

    













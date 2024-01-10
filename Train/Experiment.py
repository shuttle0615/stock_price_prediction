from Train import *

class experiment():
    def __init__(self, train_dl, validation_dl, model, train_args):
        self.train_dataloader = train_dl
        self.validation_dataloader = validation_dl
        
        self.model = model
        
        self.train_args = train_args
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.train_args.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
            factor=0.25, patience=1, threshold=0.0001, threshold_mode='rel',
            cooldown=0, min_lr=0, eps=1e-07, verbose=False)
        
        self.name = f'{self.train_args.epoch}_{self.train_args.lr}_best_model.ckpt'
    
    def run_exp(self): 
        # v_loss setting
        best_vloss = 1_000_000

        # run epoch
        for epoch in range(self.train_args.epoch):
            print('EPOCH {}:'.format(epoch))

            self.model.train()
            avg_loss = train_loop(self.model, self.optimizer, self.loss_fn, self.train_dataloader)

            self.model.eval()
            avg_v_loss = validation_loop(self.model, self.loss_fn, self.validation_dataloader)

            print('LOSS train {} valid {}'.format(avg_loss, avg_v_loss))

            if avg_v_loss < best_vloss:
                best_vloss = avg_v_loss
                torch.save(self.model.state_dict(),  result_dir / self.name)

            self.scheduler.step(avg_v_loss)
                
                
            
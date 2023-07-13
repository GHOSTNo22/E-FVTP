import pickle
import random
import time
import torch
from rich.progress import track
from tqdm import tqdm
import sys
from base import ServerBase
from client.scaffold import SCAFFOLDClient
from config.util import clone_parameters, get_args
import pandas as pd
torch.set_printoptions(precision=8)
class SCAFFOLDServer(ServerBase):
    def __init__(self):
        super(SCAFFOLDServer, self).__init__(get_args(), "SCAFFOLD")

        self.trainer = SCAFFOLDClient(
            backbone= self.backbone(self.args.dataset) if self.args.dataset != 'AIS' else self.backbone(),
            dataset=self.args.dataset,
            batch_size=self.args.batch_size,
            local_epochs=self.args.local_epochs,
            local_lr=self.args.local_lr,
            logger=self.logger,
            gpu=self.args.gpu,
        )
        backbone = self.backbone(self.args.dataset) if self.args.dataset != 'AIS' else self.backbone()
        self.c_global = [
            torch.zeros_like(param).to(self.device)
            for param in backbone.parameters()
        ]
        self.global_lr = 1 
        self.training_acc = [[] for _ in range(self.global_epochs)]

    def train(self):
        self.logger.log("=" * 30, "TRAINING", "=" * 30, style="bold green")
        start = time.perf_counter()
        progress_bar = (
            track(
                range(self.global_epochs),
                "[bold green]Training...",
                console=self.logger,
            )
            if not self.args.log
            else tqdm(range(self.global_epochs), "Training...")
        )
        com_size = 0
        out_csv = pd.DataFrame(columns=['round','Loss','time'])
        round_list = []
        loss_list = []
        time_list = []
        for E in progress_bar:

            if E % self.args.verbose_gap == 0:
                self.logger.log("=" * 30, f"ROUND: {E}", "=" * 30)

            selected_clients = random.sample(
                self.client_id_indices, self.args.client_num_per_round
            )
            res_cache = []
            com_size = com_size + len(self.client_id_indices) * sys.getsizeof(clone_parameters(self.global_params_dict))

            for client_id in selected_clients:
                client_local_params = clone_parameters(self.global_params_dict)
                res, stats = self.trainer.train(
                    client_id=client_id,
                    model_params=client_local_params,
                    c_global=self.c_global,
                    verbose=(E % self.args.verbose_gap) == 0,
                )
                res_cache.append(res)
                for x in res:
                    com_size = com_size + sys.getsizeof(x)
                com_size = com_size + sys.getsizeof(res) + sys.getsizeof(self.c_global)
                #print(sys.getsizeof(res))
                self.num_correct[E].append(stats["correct"])
                self.num_samples[E].append(stats["size"])
            #print(sys.getsizeof(res_cache))
            #com_size = com_size + sys.getsizeof(res_cache)
            self.aggregate(res_cache)
            if E % self.args.verbose_gap == 0:
                time_stamp = time.perf_counter()
                t = time_stamp - start
                self.logger.log("=" * 30, f"ROUND_time: {t}", "=" * 30)
                #        self.test()
                self.logger.log("=" * 30, "TESTING", "=" * 30, style="bold blue")
                all_loss = []
                all_correct = []
                all_samples = []
                for client_id in self.client_id_indices:
                    client_local_params = clone_parameters(self.global_params_dict)
                    stats = self.trainer.test(
                        client_id=client_id,
                        model_params=client_local_params,
                    )
                    # self.logger.log(
                    #     f"client [{client_id}]  [red]loss: {(stats['loss'] / stats['size']):.4f}    [magenta]accuracy: {stats(['correct'] / stats['size'] * 100):.2f}%"
                    # )
                    all_loss.append(stats["loss"])
                    all_correct.append(stats["correct"])
                    all_samples.append(stats["size"])
                self.logger.log("=" * 20, "RESULTS", "=" * 20, style="bold green")
                self.logger.log(
                    "loss: {:.8f}    accuracy: {:.2f}%".format(
                        sum(all_loss) / sum(all_samples),
                        sum(all_correct) / sum(all_samples) * 100.0,
                    )
                )
                round_list.append(E)
                time_list.append(t)
                loss_list.append((sum(all_loss)/sum(all_samples)).cpu())
                

            if E % self.args.save_period == 0 and self.args.save_period > 0:
                torch.save(
                    self.global_params_dict,
                    self.temp_dir / "global_model.pt",
                )
                with open(self.temp_dir / "epoch.pkl", "wb") as f:
                    pickle.dump(E, f)
        data = {'round':round_list,'time':time_list,'Loss':loss_list}
        print(data)
        out_csv = pd.DataFrame(data)
        out_csv.to_csv(r'//mnt//VMSTORE//fedves//'+self.algo+'.csv')
        avg_com_size = com_size / self.global_epochs
        print("avg_com_size:" + str(avg_com_size))
        print("com_size:" + str(com_size))

    def aggregate(self, res_cache):
        y_delta_cache = list(zip(*res_cache))[0]
        c_delta_cache = list(zip(*res_cache))[1]
        trainable_parameter = filter(
            lambda param: param.requires_grad, self.global_params_dict.values()
        )

        # update global model
        avg_weight = torch.tensor(
            [
                1 / self.args.client_num_per_round
                for _ in range(self.args.client_num_per_round)
            ],
            device=self.device,
        )
        for param, y_del in zip(trainable_parameter, zip(*y_delta_cache)):
            x_del = torch.sum(avg_weight * torch.stack(y_del, dim=-1), dim=-1)
            param.data += self.global_lr * x_del

        # update global control
        for c_g, c_del in zip(self.c_global, zip(*c_delta_cache)):
            c_del = torch.sum(avg_weight * torch.stack(c_del, dim=-1), dim=-1)
            c_g.data += (
                self.args.client_num_per_round / len(self.client_id_indices)
            ) * c_del


if __name__ == "__main__":
    server = SCAFFOLDServer()
    server.run()

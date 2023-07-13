from base import ServerBase
from client.fedavg import FedAvgClient
from config.util import get_args


class FedAvgServer(ServerBase):
    def __init__(self):
        super(FedAvgServer, self).__init__(get_args(), "FedAvg")
        self.trainer = FedAvgClient(
            backbone=self.backbone(self.args.dataset)  if self.args.dataset != 'AIS' else self.backbone(),
            dataset=self.args.dataset,
            batch_size=self.args.batch_size,
            local_epochs=self.args.local_epochs,
            local_lr=self.args.local_lr,
            logger=self.logger,
            gpu=self.args.gpu,
        )


if __name__ == "__main__":
    server = FedAvgServer()
    server.run()

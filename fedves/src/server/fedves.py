from base import ServerBase
from config.util import get_args

from fedves.src.client.fedves import FedvesClient


class FedvesServer(ServerBase):
    def __init__(self):
        super(FedvesServer, self).__init__(get_args(), "Fedves")

        self.trainer = FedvesClient(
            backbone=self.backbone(self.args.dataset) if self.args.dataset != 'AIS' else self.backbone(),
            dataset=self.args.dataset,
            batch_size=self.args.batch_size,
            local_epochs=self.args.local_epochs,
            local_lr=self.args.local_lr,
            logger=self.logger,
            gpu=self.args.gpu,
        )


if __name__ == "__main__":
    server = FedvesServer()
    server.run()

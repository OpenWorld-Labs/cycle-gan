from .cyclegan import CycleGANTrainer

def get_trainer_cls(trainer_id):
    if trainer_id == "cyclegan":
        return CycleGANTrainer
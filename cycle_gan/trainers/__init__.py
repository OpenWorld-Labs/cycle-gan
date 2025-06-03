def get_trainer_cls(trainer_id):
    if trainer_id == "cyclegan":
        from .cyclegan import CycleGANTrainer
        return CycleGANTrainer
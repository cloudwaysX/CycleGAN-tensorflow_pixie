
def create_model(opt):
    model = None
    from .cycle_gan_model import CycleGANModel
    model = CycleGANModel()
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model

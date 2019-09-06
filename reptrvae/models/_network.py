class Network(object):
    def __init__(self):
        pass

    @NotImplementedError
    def __create_network(self):
        pass

    @NotImplementedError
    def __compile_network(self):
        pass

    @NotImplementedError
    def restore_model(self):
        pass

    @NotImplementedError
    def save_model(self):
        pass

    @NotImplementedError
    def predict(self):
        pass

    @NotImplementedError
    def to_latent(self):
        pass

    @NotImplementedError
    def train(self):
        pass


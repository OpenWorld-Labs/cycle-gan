from . import local_cod

def get_loader(data_id, batch_size, **data_kwargs):
    if data_id == "local_cod":
        return local_cod.get_loader(batch_size, **data_kwargs)
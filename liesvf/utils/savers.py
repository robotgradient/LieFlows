import torch
import os



def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def save_in_torch_file(file_name, list_names, list_objects):
    makedirs(os.path.dirname(file_name))

    directory = {}
    for i in range(len(list_names)):
        directory[list_names[i]] = list_objects[i]

    torch.save(directory,file_name)

def load_torch_file(file_name):
    return torch.load(file_name)


if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    file = 'test.pth'
    file_dir = os.path.join(dirname,file)

    list_names = ['a','b','c']
    list_objects = [1,2,3]
    save_in_torch_file(file_dir, list_names, list_objects)

    x = torch.load(file_dir)
    print(x)

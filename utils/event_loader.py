from torch.utils.data import Dataset
from typing import Any
from .utils import get_file
import numpy as np
import os

class myEventLoader(Dataset):
    fields=['wall_time','step','value']
    def __init__(self, file_path) -> None:
        self.file_path = get_file(file_path, part_name="events.out.tfevents.")[0] 
        # Numbers = [str(num) for num in range(10)]
        # self.file_path = [file for file in file_path if os.path.splitext(file)[-1][-1] in Numbers]
        if self.file_path==[]:
            Warning("There're no events files in the given path")       

    def get_item(self, file_path) -> Any:
        from tensorboard.backend.event_processing import event_accumulator
        ea = event_accumulator.EventAccumulator(file_path)
        ea.Reload()
        self.Key = ea.Tags()['scalars']
        self.Data = {ea.summary_metadata[key].display_name: ea.Scalars(key) for key in self.Key}
            
        result = {tag: {key: [item.__getattribute__(key)-(data[0].wall_time if key=='wall_time' else 0) for item in data] 
                             for key in self.fields
                        }
                       for tag,data in self.Data.items()
                 }
        
        return result

    def events_to_mat(self, file_num=-1, write_file=True):    
        # call this function to convert the events files of given number in path to *.mat
        from scipy.io import savemat
        if file_num==-1:
            file_list = [num for num in range(len(self.file_path))]
        else:
            file_list = [file_num] if isinstance(file_num, int) else file_num
        mat_files = []
        for num in file_list:     
            mat_file = self.file_path[num]+".mat" 
            if os.path.exists(mat_file):
                mat_files.append(mat_file)
            else:
                if num>=len(self.file_path): 
                    continue    
                # self.readFile(file_path=self.file_path[num])
                results = self.get_item(file_path=self.file_path[num])
                mat_files.append(mat_file)
                if results=={}:
                    continue
                else:
                    if write_file:
                        savemat(mat_files[num], results)
                        print(f'event file {self.file_path[num].split(os.sep)[-1]} converted to .mat.')
        
        return mat_files
    
    def events_to_yaml(self, file_num=-1):   
        from scipy.io import loadmat
        import yaml
        if file_num==-1:
            file_list = [num for num in range(len(self.file_path))]
        else:
            file_list = [file_num] if isinstance(file_num, int) else file_num
        for num in file_list:     
            eventfile = self.file_path[num]    
            try:
                data=loadmat(eventfile+".mat")
                if not data=={}:
                    with open(eventfile+".yaml", 'w') as file:
                        yaml.dump(data, file) 
            except:
                pass

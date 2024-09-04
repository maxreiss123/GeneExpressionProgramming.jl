import os
import shutil
from datetime import datetime
import numpy as np
import subprocess


class FileManager(object):
    def __init__(self, config):
        self.start_date = datetime.now().strftime('%d%H%M%S')
        self.target_name = config.get('name', 'cfd_loop') + '_' + self.start_date
        self.running_template_path = config.get('running_template_path', 'running_template')
        self.input_template_name = config.get('input_template_name', 'input_template')
        self.eval_template_path = config.get('eval_template_path', 'eval_template')
        self.output_source = config.get('output_folder_name', 'output')
        self.cost_fun_identifier = config.get('target_path', 'C1')
        self.eval_model_file = config.get('eval_model_file', 'eval_model.py')
        self.task_folder_suffix = config.get('suffix', '*_eve_task')
        self.config = config
        self.init_folder()

    def create_cfd_instance(self, ind_id, ind_expression):
        indi_name = f"run_{ind_id}"
        indi_operating_path = os.path.join(self.target_name, indi_name)
        shutil.copytree(os.path.join(self.target_name, self.eval_template_path), indi_operating_path,
                        dirs_exist_ok=True)
        self.infer_expression_to_file(indi_operating_path, ind_expression, ind_id)
        return indi_operating_path

    def deconstruct_cfd_instance(self, target_path):
        #os.system(f'rm -rf {os.path.join(target_path, self.task_folder_suffix)}')
        os.system(f'rm -rf {target_path}')

    def execute_external_file(self, target_path):
        return subprocess.call(["python", os.path.join(target_path, self.eval_model_file)])

    def retrieve_value(self, target_path):
        if type(self.cost_fun_identifier) != list:
            return np.loadtxt(os.path.join(target_path, self.output_source, f"{self.cost_fun_identifier}.edf.gz"))
        else:
            errs = [np.loadtxt(os.path.join(target_path, self.output_source, f"{elem}.edf.gz")) for elem in
                    self.cost_fun_identifier]
            try:
                errs = [float(err) for err in errs]
            except:
                print("issue")
                print(errs)
            return errs

    def init_folder(self):
        shutil.copytree(self.running_template_path, self.target_name)
        print(f"Run is created in {self.target_name}")
        try:
            config_frame = pd.DataFrame(self.config)
            config_frame.to_csv(os.path.join(self.target_name, 'config.csv'))
        except Exception as e:
            print("Something went wrong - writing config file")

    def infer_expression_to_file(self, operating_path, ind_expression, ind_id):
        input_path_temp = os.path.join(operating_path, self.input_template_name)
        handler = open(input_path_temp, 'a')
        for expression in ind_expression:
            handler.write(f"{expression:.4f}")
            handler.write('\n')
        handler.close()
        shutil.move(input_path_temp, os.path.join(operating_path, f"input_{ind_id}"))

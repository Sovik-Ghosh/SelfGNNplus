import time
import json
import os
from Params import args
from tabulate import tabulate
import numpy as np
import tensorflow as tf

class TrainingLogger:
    def __init__(self, log_dir, hyperparameters):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.start_time = None
        self.hyperparameters = hyperparameters
        self.model_params = {}
    
    def start_training(self):
        """Start the timer for training."""
        self.start_time = time.time()
    
    def end_training(self):
        """End the timer for training and calculate total time."""
        if self.start_time is None:
            raise RuntimeError("Training not started. Call start_training() first.")
        end_time = time.time()
        total_time = end_time - self.start_time
        return total_time
    
    def save_and_print_table(self, epochNdcgs, epochHit, num, cutoffs = [5, 10, 15, 20]):
        table_data = [[f"{i}", f"{epochNdcgs[f'{i}'] / num:.4f}", f"{epochHit[f'{i}'] / num:.4f}"] for i in cutoffs]
        headers = ["Metric", "Epoch NDCG", "Epoch Hit"]
        table = tabulate(table_data, headers=headers, tablefmt="grid")
        print(table)
        with open(os.path.join(self.log_dir, "epoch_metrics.txt"), "a") as f:
            f.write(table)
            
    def makePrint(self, name, ep, reses, save, metrics):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in metrics:
                metrics[tem].append(val)
        ret = ret[:-2] + '  '
        print(ret)

        # Log to file
        with open(os.path.join(self.log_dir, "epoch_metrics.txt"), "a") as file:
            file.write('\n' +ret + '\n')
    
    def save_trainable_parameters_to_file(self, model):
        total_parameters = 0
        with open(os.path.join(self.log_dir, "parameters.txt"), 'w') as file:
            for var in model.weights:
                var_parameters = tf.reduce_prod(var.shape).numpy()
                total_parameters += var_parameters
                file.write(f'Variable: {var.name}, Shape: {var.shape}, Total parameters: {var_parameters}\n')
            for layer in model.layers:
                for variable in layer.trainable_variables:
                    shape = variable.shape
                    variable_parameters = tf.reduce_prod(shape).numpy()
                    file.write(f'Variable: {variable.name}, Shape: {shape}, Total parameters: {variable_parameters}\n')
                    total_parameters += variable_parameters
            file.write(f'Total trainable parameters: {total_parameters}\n')

    def save_config(self):
        """Save hyperparameters and training configuration."""
        with open(os.path.join(self.log_dir, "config.json"), "w") as f:
            json.dump(self.hyperparameters, f, indent=4)

    def print_summary(self, model):
        """Print the summary of training including total time."""
        total_time = self.end_training()
        with open(os.path.join(self.log_dir, "epoch_metrics.txt"), "a") as file:
            file.write(f'\nTotal time taken to train: {total_time}\n')
        print(f"Total training time: {total_time:.2f} seconds")
        print("Epoch metrics saved to epoch_metrics.txt.")
        self.save_trainable_parameters_to_file(model)
        print("Model parameters saved to model_params.json.")
        self.save_config()
        print("Training configuration saved to config.json.")
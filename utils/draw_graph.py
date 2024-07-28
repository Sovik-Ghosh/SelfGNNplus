import pandas as pd
import plotly.graph_objects as go
import os
import sys

# Add the previous directory to the system path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Params import args

def draw_eval(file_paths, name):
    """
    Draw evaluation metrics for multiple CSV files.
    
    Parameters:
    file_paths (list of str): List of paths to evaluation CSV files.
    """
    # Dictionary to store data for each metric
    metrics_data = {}

    # Iterate over each file
    for file_path in file_paths:
        data = pd.read_csv(file_path)

        # Iterate over each unique metric in the current file
        for metric in data['Metric'].unique():
            metric_data = data[data['Metric'] == metric].reset_index(drop=True)
            if metric not in metrics_data:
                metrics_data[metric] = []
            metrics_data[metric].append((metric_data, file_path))

    # Iterate over each metric and plot HitRate and NDCG
    for metric, metric_data_list in metrics_data.items():
        fig = go.Figure()

        # Iterate over each dataset for the current metric
        for metric_data, file_path in metric_data_list:
            new_index = [1] + [i * args.tstEpoch for i in range(1, len(metric_data))]
            
            # Ensure new_index is the same length as metric_data
            if len(new_index) > len(metric_data):
                new_index = new_index[:len(metric_data)]

            # Add HitRate trace
            fig.add_trace(go.Scatter(x=new_index, y=metric_data['HitRate'], mode='lines+markers', name=f'HitRate {metric} - {os.path.basename(os.path.dirname(file_path))}'))

            # Add NDCG trace
            fig.add_trace(go.Scatter(x=new_index, y=metric_data['NDCG'], mode='lines+markers', name=f'NDCG {metric} - {os.path.basename(os.path.dirname(file_path))}'))

            # Mark the highest points for HitRate and NDCG
            max_hitrate = metric_data['HitRate'].max()
            max_ndcg = metric_data['NDCG'].max()
            max_hitrate_epoch = new_index[metric_data['HitRate'].idxmax()]
            max_ndcg_epoch = new_index[metric_data['NDCG'].idxmax()]

            fig.add_trace(go.Scatter(x=[max_hitrate_epoch], y=[max_hitrate], mode='markers+text', text=[f'Max {max_hitrate:.4f}'], textposition='top right', name=f'Max HitRate {metric} - {os.path.basename(os.path.dirname(file_path))}'))
            fig.add_trace(go.Scatter(x=[max_ndcg_epoch], y=[max_ndcg], mode='markers+text', text=[f'Max {max_ndcg:.4f}'], textposition='top right', name=f'Max NDCG {metric} - {os.path.basename(os.path.dirname(file_path))}'))

        # Update the layout
        fig.update_layout(
            title=f'Metrics: HitRate and NDCG for Cutoff {metric}',
            xaxis_title='Epoch',
            yaxis_title='Values',
            legend_title='Metrics',
            width = 2000,
            height = 800
        )

        dir_path = f"../output/{name}"
        create_dir_if_not_exists(dir_path)
        # Save the plot as an SVG file
        fig.write_image(f"{dir_path}/combined_plot_cutoff_{metric}.png")
        fig.show()

def draw_train(file_paths, name):
    """
    Draw training metrics for multiple CSV files.
    
    Parameters:
    file_paths (list of str): List of paths to training CSV files.
    """
    # Initialize the Plotly figure
    fig = go.Figure()

    # Iterate over each file
    for file_path in file_paths:
        data = pd.read_csv(file_path)

        # Add Loss trace
        fig.add_trace(go.Scatter(x=data['Epoch'], y=data['Loss'], mode='lines+markers', name=f'Loss - {os.path.basename(os.path.dirname(file_path))}'))

        # Add preLoss trace
        fig.add_trace(go.Scatter(x=data['Epoch'], y=data['preLoss'], mode='lines+markers', name=f'preLoss - {os.path.basename(os.path.dirname(file_path))}'))

    # Update the layout
    fig.update_layout(
        title='Loss and preLoss Over Epochs',
        xaxis_title='Epoch',
        yaxis_title='Value',
        legend_title='Metrics',
        width = 2000,
        height = 800
    )
    dir_path = f"../output/{name}"
    create_dir_if_not_exists(dir_path)

    # Save the plot as an SVG file
    fig.write_image(f"{dir_path}/loss_plot.png")
    fig.show()

def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ == '__main__':
    '''eval_file_paths = [
        '/home/sovik/SelfGNNplus/logdir/gowalla_gru_64/evaluation.csv',
        '/home/sovik/SelfGNNplus/logdir/gowalla_lstm_64/evaluation.csv',
        '/home/sovik/SelfGNNplus/logdir/gowalla_tcn_64/evaluation.csv'
    ]
    
    train_file_paths = [
        '/home/sovik/SelfGNNplus/logdir/gowalla_gru_64/training.csv',
        '/home/sovik/SelfGNNplus/logdir/gowalla_lstm_64/training.csv',
        '/home/sovik/SelfGNNplus/logdir/gowalla_tcn_64/training.csv'
    ]
    
    
    draw_eval(eval_file_paths, 'gowalla')
    draw_train(train_file_paths, 'gowalla')

    eval_file_paths = [
        '/home/sovik/SelfGNNplus/logdir/amazon_gru_64/evaluation.csv',
        '/home/sovik/SelfGNNplus/logdir/amazon_lstm_64/evaluation.csv',
        '/home/sovik/SelfGNNplus/logdir/amazon_tcn_64/evaluation.csv'
    ]
    
    train_file_paths = [
        '/home/sovik/SelfGNNplus/logdir/amazon_gru_64/training.csv',
        '/home/sovik/SelfGNNplus/logdir/amazon_lstm_64/training.csv',
        '/home/sovik/SelfGNNplus/logdir/amazon_tcn_64/training.csv'
    ]
    
    
    draw_eval(eval_file_paths, 'amazon')
    draw_train(train_file_paths, 'amazon')

    eval_file_paths = [
        '/home/sovik/SelfGNNplus/logdir/yelp_gru_64/evaluation.csv',
        '/home/sovik/SelfGNNplus/logdir/yelp_lstm_64/evaluation.csv',
        '/home/sovik/SelfGNNplus/logdir/yelp_tcn_64/evaluation.csv'
    ]
    
    train_file_paths = [
        '/home/sovik/SelfGNNplus/logdir/yelp_gru_64/training.csv',
        '/home/sovik/SelfGNNplus/logdir/yelp_lstm_64/training.csv',
        '/home/sovik/SelfGNNplus/logdir/yelp_tcn_64/training.csv'
    ]
    
    
    draw_eval(eval_file_paths, 'yelp')
    draw_train(train_file_paths, 'yelp')'''

    eval_file_paths = [
        '/home/sovik/SelfGNNplus/logdir/movielens_gru_64/evaluation.csv',
        '/home/sovik/SelfGNNplus/logdir/movielens_lstm_64/evaluation.csv',
        '/home/sovik/SelfGNNplus/logdir/movielens_tcn_64/evaluation.csv'
    ]
    
    train_file_paths = [
        '/home/sovik/SelfGNNplus/logdir/movielens_gru_64/training.csv',
        '/home/sovik/SelfGNNplus/logdir/movielens_lstm_64/training.csv',
        '/home/sovik/SelfGNNplus/logdir/movielens_tcn_64/training.csv'
    ]
    
    
    draw_eval(eval_file_paths, 'movielens')
    draw_train(train_file_paths, 'movielens')

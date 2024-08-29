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
    Draw evaluation metrics for multiple CSV files with HR and NDCG in one graph,
    larger text, alternating text annotations, and legend at the bottom.
    
    Parameters:
    file_paths (list of str): List of paths to evaluation CSV files.
    name (str): Name used for output directory and file naming.
    """
    metrics_data = {}

    for file_path in file_paths:
        data = pd.read_csv(file_path)
        for metric in data['Metric'].unique():
            metric_data = data[data['Metric'] == metric].reset_index(drop=True)
            if metric not in metrics_data:
                metrics_data[metric] = []
            metrics_data[metric].append((metric_data, file_path))

    for metric, metric_data_list in metrics_data.items():
        fig = go.Figure()

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for idx, (metric_data, file_path) in enumerate(metric_data_list):
            new_index = [1] + [i * args.tstEpoch for i in range(1, len(metric_data))]
            if len(new_index) > len(metric_data):
                new_index = new_index[:len(metric_data)]

            model_name = os.path.basename(os.path.dirname(file_path))
            color = colors[idx % len(colors)]

            # HitRate
            fig.add_trace(go.Scatter(x=new_index, y=metric_data['HitRate'], mode='lines+markers', 
                                     name=f'HitRate - {model_name}', line=dict(color=color, width=4), 
                                     marker=dict(size=16, symbol='circle')))

            # NDCG
            fig.add_trace(go.Scatter(x=new_index, y=metric_data['NDCG'], mode='lines+markers', 
                                     name=f'NDCG - {model_name}', line=dict(color=color, dash='dash', width=4), 
                                     marker=dict(size=16, symbol='square')))

            # Mark max points
            max_hitrate = metric_data['HitRate'].max()
            max_ndcg = metric_data['NDCG'].max()
            max_hitrate_epoch = new_index[metric_data['HitRate'].idxmax()]
            max_ndcg_epoch = new_index[metric_data['NDCG'].idxmax()]

            fig.add_trace(go.Scatter(x=[max_hitrate_epoch], y=[max_hitrate], mode='markers', 
                                     marker=dict(size=24, symbol='star', color=color), 
                                     showlegend=False))
            fig.add_trace(go.Scatter(x=[max_ndcg_epoch], y=[max_ndcg], mode='markers', 
                                     marker=dict(size=24, symbol='star-triangle-up', color=color), 
                                     showlegend=False))

            # Add text annotations with alternating positions
            hr_yshift = 60 if idx % 2 == 0 else -60
            ndcg_yshift = -60 if idx % 2 == 0 else 60

            fig.add_annotation(x=max_hitrate_epoch, y=max_hitrate,
                               text=f'Max HR: {max_hitrate:.4f}',
                               showarrow=False,
                               yshift=hr_yshift,
                               font=dict(size=32, color=color))
            fig.add_annotation(x=max_ndcg_epoch, y=max_ndcg,
                               text=f'Max NDCG: {max_ndcg:.4f}',
                               showarrow=False,
                               yshift=ndcg_yshift,
                               font=dict(size=32, color=color))

        fig.update_layout(
            title=dict(text=f'HitRate and NDCG for Cutoff {metric}', font=dict(size=64)),
            xaxis_title=dict(text='Epoch', font=dict(size=48)),
            yaxis_title=dict(text='Value', font=dict(size=48)),
            legend=dict(font=dict(size=36), orientation='h', yanchor='top', y=-0.1, xanchor='center', x=0.5),
            width=2400,
            height=1600,  # Increased height to accommodate larger text and bottom legend
            template='plotly_white',
            margin=dict(b=250)  # Increased bottom margin for legend
        )

        fig.update_xaxes(tickfont=dict(size=36))
        fig.update_yaxes(tickfont=dict(size=36))

        # Add a rectangle to highlight the best performance area
        y_range = fig.layout.yaxis.range
        if y_range is None:  # If range is not set, calculate it
            y_values = [trace.y for trace in fig.data if trace.y is not None]
            y_range = [min(min(y) for y in y_values), max(max(y) for y in y_values)]
        
        fig.add_shape(type="rect",
            x0=max(max_hitrate_epoch, max_ndcg_epoch) - 5, x1=max(max_hitrate_epoch, max_ndcg_epoch) + 5,
            y0=y_range[0], y1=y_range[1],
            fillcolor="LightSalmon", opacity=0.3, layer="below", line_width=0,
        )

        dir_path = f"../output/{name}"
        create_dir_if_not_exists(dir_path)
        fig.write_image(f"{dir_path}/combined_plot_cutoff_{metric}.png", scale=2)

def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def cap_name(name):
    n = name[21:23]
    if n=='5.':
        return 'Cutoff-5'
    elif n=='10':
        return 'Cutoff-10'
    elif n=='15':
        return 'Cutoff-15'
    elif n=='20':
        return 'Cutoff-20'
    else:
        return None

def escape_latex(text):
    """Escapes special LaTeX characters in the text."""
    # Dictionary of replacements to be applied from left to right
    replacements = {
        '\\': r'\textbackslash{}',
        '{': r'\{',
        '}': r'\}',
        '_': r'\_',
        '^': r'\^',
        '&': r'\&',
        '%': r'\%',
        '$': r'\$',
        '~': r'\~{}',
        '#': r'\#'
    }
    
    # Perform replacements in order to avoid issues with already escaped text
    for search, replace in replacements.items():
        text = text.replace(search, replace)
    
    return text

def generate_latex_for_images(image_folder, output_tex_file):
    with open(output_tex_file, 'w') as tex_file:
        tex_file.write("\\appendix\n")
        tex_file.write("\\section{Appendix: Additional Figures}\n\n")

        # Iterate over each subdirectory in the specified image_folder
        for subdir in os.listdir(image_folder):
            subdir_path = os.path.join(image_folder, subdir)

            if os.path.isdir(subdir_path):
                # Get a list of all files in the subdirectory
                images = [f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))]

                # Filter only image files
                image_extensions = ['.png', '.jpg', '.jpeg', '.pdf', '.svg']
                images = [img for img in images if os.path.splitext(img)[1].lower() in image_extensions]

                if images:
                    tex_file.write(f"\\section{{Figures from {escape_latex(subdir)}}}\n\n")
                    tex_file.write("\\begin{figure}[H]\n")
                    tex_file.write("    \\centering\n")

                    # Define the width for subfigures
                    num_images = len(images)
                    width = 0.45

                    for img in images:
                        # Remove extension for label

                        name = cap_name(img)
                        if name is None:
                            continue
                        label = os.path.splitext(img)[0].replace(' ', '_')
                        tex_file.write(f"    \\subfigure[{name}]{{\n")
                        tex_file.write(f"        \\includegraphics[width={width:.2f}\\textwidth]{{output/{subdir}/{img}}}\n")
                        tex_file.write(f"        \\label{{fig:{label}}}\n")
                        tex_file.write("    }\n")

                    tex_file.write("\\end{figure}\n\n")

    print(f"LaTeX code written to {output_tex_file}")

if __name__ == '__main__':
    '''eval_file_paths = [
        '/home/sovik/SelfGNNplus/logdir/yelp_64/evaluation.csv',
        '/home/sovik/SelfGNNplus/logdir/yelp_tcn_64/evaluation.csv'
    ]
    draw_eval(eval_file_paths, 'yelpN')

    eval_file_paths = [
        '/home/sovik/SelfGNNplus/logdir/yelp_nolong/evaluation.csv',
        '/home/sovik/SelfGNNplus/logdir/yelp_tcn_64/evaluation.csv'
    ]
    draw_eval(eval_file_paths, 'yelp_nolong')

    eval_file_paths = [
        '/home/sovik/SelfGNNplus/logdir/yelp_nossl/evaluation.csv',
        '/home/sovik/SelfGNNplus/logdir/yelp_tcn_64/evaluation.csv'
    ]
    draw_eval(eval_file_paths, 'yelp_nossl')
    eval_file_paths = [
        '/home/sovik/SelfGNNplus/logdir/yelp_gru_64/evaluation.csv',
        '/home/sovik/SelfGNNplus/logdir/yelp_tcn_64/evaluation.csv',
        '/home/sovik/SelfGNNplus/logdir/yelp_lstm_64/evaluation.csv'
    ]
    draw_eval(eval_file_paths, 'yelp')
    eval_file_paths = [
        '/home/sovik/SelfGNNplus/logdir/amazon_gru_64/evaluation.csv',
        '/home/sovik/SelfGNNplus/logdir/amazon_tcn_64/evaluation.csv',
        '/home/sovik/SelfGNNplus/logdir/amazon_lstm_64/evaluation.csv'
    ]
    draw_eval(eval_file_paths, 'amazon')
    eval_file_paths = [
        '/home/sovik/SelfGNNplus/logdir/gowalla_gru_64/evaluation.csv',
        '/home/sovik/SelfGNNplus/logdir/gowalla_tcn_64/evaluation.csv',
        '/home/sovik/SelfGNNplus/logdir/gowalla_lstm_64/evaluation.csv'
    ]
    draw_eval(eval_file_paths, 'gowalla')'''
    eval_file_paths = [
        '/home/sovik/SelfGNNplus/logdir/amazon_none/evaluation.csv',
        '/home/sovik/SelfGNNplus/logdir/amazon_tcn_64/evaluation.csv',
    ]
    draw_eval(eval_file_paths, 'amazon_none')



    # Specify the folder where your images are stored and the output .tex file name
    #image_folder = '/home/sovik/SelfGNNplus/output'
    #output_tex_file = 'generated_appendix.tex'

    
    #generate_latex_for_images(image_folder, output_tex_file)

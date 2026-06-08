import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
from prettytable import PrettyTable
from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info




def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'


if __name__ == '__main__':
    model_path = r'ultralytics-main\runs\detect\MEG-YOLO\weights\best.pt'
    model = YOLO(model_path)  
    result = model.val(data=r'E:\anaconda\project\yolov12\ultralytics-main\ultralytics\cfg\datasets\5.yaml',
                       split='val',  
                       imgsz=1280,
                       batch=4,
                       conf=0.001,
                       # iou=0.7,
                       # rect=False,
                       save_json=True,  # if you need to cal coco metrice
                       project='runs/val',
                       name='exp',
                       )

    if model.task == 'detect':  

        try:
            
            if len(result.box.curves_results) > 2:
                roc_data = result.box.curves_results[2] 
                save_data = {
                    'fpr': roc_data[0],
                    'tpr': roc_data[1],
                    'scores': roc_data[2],
                    'model_name': 'YOLOv12'
                }
            else:
                
                pr_data = result.box.curves_results[0]
                save_data = {
                    'fpr': 1 - pr_data[1],
                    'tpr': pr_data[0],
                    'scores': pr_data[2],
                    'model_name': 'YOLOv8'
                }

            np.save(os.path.join(result.save_dir, 'yolo_roc_data.npy'), save_data)
            print(f'>>> 修正后的 ROC 数据已保存至: {result.save_dir}/yolo_roc_data.npy')

        except Exception as e:
           
            print(f"提取曲线数据失败: {e}")

            
        length = result.box.p.size
        model_names = list(result.names.values())
        preprocess_time_per_image = result.speed['preprocess']

        length = result.box.p.size
        model_names = list(result.names.values())
        preprocess_time_per_image = result.speed['preprocess']
        inference_time_per_image = result.speed['inference']
        postprocess_time_per_image = result.speed['postprocess']
        all_time_per_image = preprocess_time_per_image + inference_time_per_image + postprocess_time_per_image

        n_l, n_p, n_g, flops = model_info(model.model)

        print('-' * 20 + '论文上的数据以以下结果为准' + '-' * 20)
        print('-' * 20 + '论文上的数据以以下结果为准' + '-' * 20)
        print('-' * 20 + '论文上的数据以以下结果为准' + '-' * 20)
        print('-' * 20 + '论文上的数据以以下结果为准' + '-' * 20)
        print('-' * 20 + '论文上的数据以以下结果为准' + '-' * 20)

        model_info_table = PrettyTable()
        model_info_table.title = "Model Info"
        model_info_table.field_names = ["GFLOPs", "Parameters", "前处理时间/一张图", "推理时间/一张图",
                                        "后处理时间/一张图", "FPS(前处理+模型推理+后处理)", "FPS(推理)",
                                        "Model File Size"]
        model_info_table.add_row([f'{flops:.1f}', f'{n_p:,}',
                                  f'{preprocess_time_per_image / 1000:.6f}s', f'{inference_time_per_image / 1000:.6f}s',
                                  f'{postprocess_time_per_image / 1000:.6f}s', f'{1000 / all_time_per_image:.2f}',
                                  f'{1000 / inference_time_per_image:.2f}', f'{get_weight_size(model_path)}MB'])
        print(model_info_table)

        model_metrice_table = PrettyTable()
        model_metrice_table.title = "Model Metrice"
        model_metrice_table.field_names = ["Class Name", "Precision", "Recall", "F1-Score", "mAP50", "mAP75",
                                           "mAP50-95"]
        for idx in range(length):
            model_metrice_table.add_row([
                model_names[idx],
                f"{result.box.p[idx]:.4f}",
                f"{result.box.r[idx]:.4f}",
                f"{result.box.f1[idx]:.4f}",
                f"{result.box.ap50[idx]:.4f}",
                f"{result.box.all_ap[idx, 5]:.4f}",  # 50 55 60 65 70 75 80 85 90 95
                f"{result.box.ap[idx]:.4f}"
            ])
        model_metrice_table.add_row([
            "all(平均数据)",
            f"{result.results_dict['metrics/precision(B)']:.4f}",
            f"{result.results_dict['metrics/recall(B)']:.4f}",
            f"{np.mean(result.box.f1[:length]):.4f}",
            f"{result.results_dict['metrics/mAP50(B)']:.4f}",
            f"{np.mean(result.box.all_ap[:length, 5]):.4f}",  # 50 55 60 65 70 75 80 85 90 95
            f"{result.results_dict['metrics/mAP50-95(B)']:.4f}"
        ])
        print(model_metrice_table)

        with open(result.save_dir / 'paper_data.txt', 'w+') as f:
            f.write(str(model_info_table))
            f.write('\n')
            f.write(str(model_metrice_table))

        print('-' * 20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-' * 20)
        print('-' * 20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-' * 20)
        print('-' * 20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-' * 20)
        print('-' * 20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-' * 20)
        print('-' * 20, f'结果已保存至{result.save_dir}/paper_data.txt...', '-' * 20)

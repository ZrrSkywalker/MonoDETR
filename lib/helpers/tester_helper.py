import os
import tqdm
import shutil

import torch
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections


class Tester(object):
    def __init__(self, cfg, model, dataloader, logger, train_cfg=None, model_name='monodetr'):
        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        self.max_objs = dataloader.dataset.max_objs    # max objects per images, defined in dataset
        self.class_name = dataloader.dataset.class_name
        self.output_dir = os.path.join('./' + train_cfg['save_path'], model_name)
        self.dataset_type = cfg.get('type', 'KITTI')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.train_cfg = train_cfg
        self.model_name = model_name

    def test(self):
        assert self.cfg['mode'] in ['single', 'all']

        # test a single checkpoint
        if self.cfg['mode'] == 'single' or not self.train_cfg["save_all"]:
            if self.train_cfg["save_all"]:
                checkpoint_path = os.path.join(self.output_dir, "checkpoint_epoch_{}.pth".format(self.cfg['checkpoint']))
            else:
                checkpoint_path = os.path.join(self.output_dir, "checkpoint_best.pth")
            assert os.path.exists(checkpoint_path), f'{checkpoint_path}'
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=checkpoint_path,
                            map_location=self.device,
                            logger=self.logger)
            self.model.to(self.device)
            self.inference()
            self.evaluate()

        # test all checkpoints in the given dir
        elif self.cfg['mode'] == 'all' and self.train_cfg["save_all"]:
            start_epoch = int(self.cfg['checkpoint'])
            checkpoints_list = []
            for _, _, files in os.walk(self.output_dir):
                for f in files:
                    if f.endswith(".pth") and int(f[17:-4]) >= start_epoch:
                        checkpoints_list.append(os.path.join(self.output_dir, f))
            checkpoints_list.sort(key=os.path.getmtime)

            for checkpoint in checkpoints_list:
                load_checkpoint(model=self.model,
                                optimizer=None,
                                filename=checkpoint,
                                map_location=self.device,
                                logger=self.logger)
                self.model.to(self.device)
                self.inference()
                self.evaluate()

    def inference(self):
        torch.set_grad_enabled(False)
        self.model.eval()

        results = {}
        progress_bar = tqdm.tqdm(total=len(self.dataloader), leave=True, desc='Evaluation Progress')
        model_infer_time = 0
        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.dataloader):
            # load evaluation data and move data to GPU.
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            img_sizes = info['img_size'].to(self.device)

            start_time = time.time()
            ###dn
            outputs = self.model(inputs, calibs, targets, img_sizes, dn_args = 0)
            ###
            end_time = time.time()
            model_infer_time += end_time - start_time

            dets = extract_dets_from_outputs(outputs=outputs, K=self.max_objs, topk=self.cfg['topk'])

            dets = dets.detach().cpu().numpy()

            # get corresponding calibs & transform tensor to numpy
            calibs = [self.dataloader.dataset.get_calib(index) for index in info['img_id']]
            info = {key: val.detach().cpu().numpy() for key, val in info.items()}
            cls_mean_size = self.dataloader.dataset.cls_mean_size
            dets = decode_detections(
                dets=dets,
                info=info,
                calibs=calibs,
                cls_mean_size=cls_mean_size,
                threshold=self.cfg.get('threshold', 0.2))

            results.update(dets)
            progress_bar.update()

        print("inference on {} images by {}/per image".format(
            len(self.dataloader), model_infer_time / len(self.dataloader)))

        progress_bar.close()

        # save the result for evaluation.
        self.logger.info('==> Saving ...')
        self.save_results(results)
        # Save plotted results
        self.save_results_plot(results)


    def save_results(self, results):
        output_dir = os.path.join(self.output_dir, 'outputs', 'data')
        os.makedirs(output_dir, exist_ok=True)

        for img_id in results.keys():
            if self.dataset_type == 'KITTI':
                output_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id))
            else:
                os.makedirs(os.path.join(output_dir, self.dataloader.dataset.get_sensor_modality(img_id)), exist_ok=True)
                output_path = os.path.join(output_dir,
                                           self.dataloader.dataset.get_sensor_modality(img_id),
                                           self.dataloader.dataset.get_sample_token(img_id) + '.txt')

            f = open(output_path, 'w')
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()
            
    def save_results_plot(self, results):
        """
        Save plots of detection results on images.
        Args:
            results: Dictionary of detections, keyed by image ID.
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        output_dir = os.path.join(self.output_dir, 'outputs', 'plots')
        os.makedirs(output_dir, exist_ok=True)

        for img_id, detections in results.items():
            # Fetch the original image
            img = self.dataloader.dataset.get_image(img_id)
            if img is None:
                continue

            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.imshow(img)

            for det in detections:
                class_name = self.class_name[int(det[0])]
                x_min, y_min, x_max, y_max = det[2:6]  # Extract 2D bounding box coordinates
                score = det[-1]  # Extract the detection score

                # Add bounding box
                rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                         linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)

                # Add label
                label = f"{class_name} ({score:.2f})"
                ax.text(x_min, y_min - 5, label, color='yellow', fontsize=10,
                        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

            plt.axis('off')
            output_path = os.path.join(output_dir, '{:06d}.png'.format(img_id))
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)


    def evaluate(self):
        results_dir = os.path.join(self.output_dir, 'outputs', 'data')
        assert os.path.exists(results_dir)
        result = self.dataloader.dataset.eval(results_dir=results_dir, logger=self.logger)
        return result

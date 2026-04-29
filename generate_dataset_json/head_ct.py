import os
import json

class MpddSolver(object):
    CLSNAMES = ['brain']

    def __init__(self, root='/root/autodl-tmp/AnomalyCLIP-main/dataset/HeadCT'):
        self.root = root
        # json 会生成在 HeadCT/meta.json
        self.meta_path = f'{root}/meta.json'

    def run(self):
        info = dict(train={}, test={})
        anomaly_samples = 0
        normal_samples = 0
        
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}/{cls_name}'
            for phase in ['test']:
                cls_info = []
                phase_dir = f'{cls_dir}/{phase}'
                
                # 安全检查：确认文件夹是否存在
                if not os.path.exists(phase_dir):
                    print(f"❌ 找不到文件夹: {phase_dir}，请检查路径。")
                    return

                species = os.listdir(phase_dir)
                for specie in species:
                    specie_dir = f'{phase_dir}/{specie}'
                    # 过滤掉可能存在的隐藏文件，只处理文件夹
                    if not os.path.isdir(specie_dir):
                        continue
                        
                    is_abnormal = True if specie not in ['good'] else False
                    # 只读取图片文件
                    img_names = [f for f in os.listdir(specie_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
                    img_names.sort()

                    for idx, img_name in enumerate(img_names):
                        info_img = dict(
                            img_path=f'{cls_name}/{phase}/{specie}/{img_name}',
                            mask_path="",  # HeadCT 没有掩膜，保持为空
                            cls_name=cls_name,
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                        )
                        cls_info.append(info_img)
                        if phase == 'test':
                            if is_abnormal:
                                anomaly_samples += 1
                            else:
                                normal_samples += 1
                info[phase][cls_name] = cls_info
                
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")
            
        print(f"✅ 转换成功！")
        print(f"-> meta.json 已生成在: {self.meta_path}")
        print(f"-> 正常样本 (good): {normal_samples} | 异常样本 (defect): {anomaly_samples}")

if __name__ == '__main__':
    # 【已修复】加上了绝对路径 /root/...
    runner = MpddSolver(root='/root/autodl-tmp/AnomalyCLIP-main/dataset/HeadCT')
    runner.run()
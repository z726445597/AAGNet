try:
    from ijson.backends import yajl2_c as ijson
    print('using C backend')
except ImportError:
    import ijson
    print('using pure-python backend')
import json, shutil, os

SRC = r'D:\ShortEssay\Datasets\MFCAD2'
DST = r'D:\ShortEssay\Datasets\MFCAD2_smoke'
N_TRAIN, N_VAL, N_TEST = 300, 100, 100

def read_list(name):
    with open(os.path.join(SRC, name), encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]

sub = {'train': read_list('train.txt')[:N_TRAIN],
       'val':   read_list('val.txt')[:N_VAL],
       'test':  read_list('test.txt')[:N_TEST]}
keep = set(sum(sub.values(), []))
print(f'target: {len(keep)} graphs')

os.makedirs(os.path.join(DST, 'aag'), exist_ok=True)
os.makedirs(os.path.join(DST, 'labels'), exist_ok=True)

# 流式提取子集（恒定内存，约 20-60 分钟，取决于磁盘和 C 后端）
found = 0
with open(os.path.join(SRC, 'aag', 'graphs.json'), 'rb') as f, \
     open(os.path.join(DST, 'aag', 'graphs.json'), 'w', encoding='utf-8') as out:
    out.write('[\n')
    first = True
    for item in ijson.items(f, 'item'):
        if item[0] in keep:
            if not first:
                out.write(',\n')
            json.dump(item, out, default=float)
            first = False
            found += 1
            if found % 50 == 0:
                print(f'extracted {found}/{len(keep)}')
    out.write('\n]')
print(f'done: {found} graphs')

# 三个 split 清单
for split, names in sub.items():
    with open(os.path.join(DST, f'{split}.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(names) + '\n')

# 归一化统计量
shutil.copy(os.path.join(SRC, 'aag', 'attr_stat.json'),
            os.path.join(DST, 'aag', 'attr_stat.json'))

# 对应标签
missing = 0
for fn in keep:
    s = os.path.join(SRC, 'labels', fn + '.json')
    if os.path.exists(s):
        shutil.copy(s, os.path.join(DST, 'labels', fn + '.json'))
    else:
        missing += 1
print(f'missing labels: {missing}')
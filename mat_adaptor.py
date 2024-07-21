import typer
import scipy.io as scio
import numpy as np
import h5py

app = typer.Typer()

def get_circule(r):
    plus = []
    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            if i ** 2 + j ** 2 <= r ** 2:
                plus.append([i, j])
    num_one_defect = len(plus)
    print('num_one_defect:', num_one_defect)
    plus_array = np.array(plus)
    return (num_one_defect, plus, plus_array)

def get_sound():
    sound_area = []
    point9 = ([28, 171], [99, 171], [169, 173], [29, 101], [99, 101], [170, 103], [31, 30], [101, 31], [171, 32])
    for i in point9:
        for j in range(5):
            sound_area.append([i[0] + 18 + j, i[1]])
            sound_area.append([i[0] - 18 - j, i[1]])
            sound_area.append([i[0], i[1] + 18 + j])
            sound_area.append([i[0], i[1] - 18 - j])

    sound_away = (
        [[67, 8], [67, 35], [67, 71], [67, 108], [67, 143], [67, 182], [132, 13], [132, 40], [132, 74], [132, 109],
         [132, 142], [132, 179]])

    for i in sound_away:
        for j in range(-8, 10):
            for k in range(-8, 10):
                sound_area.append([i[0] + j, i[1] + k])
    print('sound_area:', len(sound_area))
    return sound_area

def get_train_labels(raw_datas, points, labels, R):
    numOneDefect, _, plus_array = get_circule(R)
    my_datas = []
    my_labels = []

    for i in range(len(points)):
        center_point = points[i]
        d = center_point + plus_array
        d_list = d.tolist()
        for j in d_list:
            my_datas.append([j[0], j[1]])
            my_labels.append(labels[str(i)])
    print('defect_all:', len(my_datas))

    sound_area = get_sound()
    for i in sound_area:
        my_datas.append([i[0], i[1]])
        my_labels.append(labels[str(9)])

    length = len(my_datas)
    print('data_all:', length)
    datas_array = np.zeros([length, raw_datas.shape[2]])
    for i in range(length):
        point = my_datas[i]
        datas_array[i, :] = raw_datas[point[0], point[1], :]
    datas_array = datas_array.astype(np.float32)
    labels_array = np.array(my_labels).astype(np.float32)
    return datas_array, labels_array

@app.command()
def process_mat_file(mat_file_path: str, output_hdf5_path: str):
    r = 10
    point9 = ([28, 171], [99, 171], [169, 173], [29, 101], [99, 101], [170, 103], [31, 30], [101, 31], [171, 32])
    label9 = {'0': 2.3, '1': 2.6, '2': 2.9, '3': 3.1, '4': 3.3, '5': 3.5, '6': 3.8, '7': 4.1, '8': 4.4, '9': 5}

    data = scio.loadmat(mat_file_path)
    data = data['AData']
    
    datas, labels = get_train_labels(data, point9, label9, r)
    
    with h5py.File(output_hdf5_path, 'w') as hf:
        # 保存数据
        dt_wave = h5py.vlen_dtype(np.float32)
        dt_label = np.float32
        
        wave = hf.create_dataset('wave',(len(datas),), dtype=dt_wave)
        
        for i, data in enumerate(datas):
            wave[i] = datas[i]
            
        hf.create_dataset('label',len(labels),data=labels, dtype=dt_label)

        # 打印结构
        print(hf.keys())

if __name__ == "__main__":
    app()

import os
import datetime
import time
import pydicom
import csv

''' ---------- Path Config ---------- '''

class workspace:
    def __init__(self, base_path, keep_prob):
        self.base_path = base_path
        self.tensorboard_path = os.path.join(base_path, 'tensorboard/')
        self.ckpt_path = os.path.join(base_path, 'ckpt_kp_%.1f/' % keep_prob)
        self.npy_path = os.path.join(base_path, 'NPY/')

# data_path = 'E:/Andream/Lung/Data/LIDC-IDRI/DOI/'
data_path = '/data0/LIDC/DOI/'

idscan_info_path = 'files/id_scan.txt'
nodule_info_path = 'files/malignancy.csv'
log_path = 'log/%s.txt' % datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
err_path = 'log/error.txt'

@property
def base_dir():
    return base_path

@base_dir.setter
def base_dir(path):
    global base_path, tensorboard_path, ckpt_path, npy_path
    base_path = path



''' ---------- File Tools ---------- '''

def mkdirs(directory):
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def open_with_mkdir(path, mode):
    directory = path[0:path.rindex('/')]
    mkdirs(directory)
    return open(path, mode)

def read_csv(filename):
    lines = []
    with open(filename, "r") as f:
        csvreader = csv.reader(f)
        for line in csvreader:
            lines.append(line)
    return lines


idscan_info = read_csv(idscan_info_path)


''' ---------- Log Tools ---------- '''

log_file = 0
def log(msg):
    global log_file
    if not log_file:
        log_file = open_with_mkdir(log_path, 'w')
    print(msg)
    log_file.write(str(msg))
    log_file.write('\n')
    log_file.flush()


err_file = 0
def err(msg):
    global err_file
    if not err_file:
        err_file = open_with_mkdir(err_path, 'a')
    print(msg)
    err_file.write(str(msg))
    err_file.write('\n')
    err_file.flush()

def logtime(msg):
    log(msg + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    return time.time()


''' ---------- malignancy.csv Tools ---------- '''

class nodule:
    def __init__(self, id, caseid, scanid, x_loc, y_loc, z_loc, sphercity, margin, lobulation, spiculation, maligancy):
        self.id, self.caseid, self.scanid = id, caseid, scanid
        self.x_loc, self.y_loc, self.z_loc = x_loc, y_loc, z_loc
        self.sphercity, self.margin, self.lobulation, self.spiculation = sphercity, margin, lobulation, spiculation
        self.maligancy = maligancy
        for idscan in idscan_info:
            if scanid in idscan[0]:
                self.scan_path = idscan[0]
                break

    def get_all_slices(self, basedir):
        slices = []
        scan_filepaths = os.listdir(basedir + self.scan_path)
        scan_filepaths.sort()
        for filepath in scan_filepaths:
            if '.dcm' in filepath:
                slices.append(pydicom.dcmread(basedir + self.scan_path + '/' + filepath))
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]), reverse=True)
        return slices

    def __str__(self):
        return 'nodule id = %s, scanid = %s, x = %d, y = %d, z = %d, malignancy = %.2f' % \
               (self.id, self.scanid,self.x_loc, self.y_loc, self.z_loc, self.maligancy)


def to_scanid(caseid):
    return 'LIDC-IDRI-%04d' % int(caseid)


def get_nodules():
    """
    获取malignancy.csv 中的结节信息，整理为nodule对象
    包括id, caseid, scanid, 结节位置信息x, y, z, 结节形态信息，以及结节恶性程度
    :return: list<nodule>
    """
    nodule_info = read_csv(nodule_info_path)
    nodules = []
    for row in nodule_info:
        nodules.append(
            nodule(
                id=row[0], caseid=row[1], scanid=to_scanid(row[1]),
                x_loc=int(row[6]), y_loc=int(row[7]), z_loc=int(row[8]),
                sphercity=float(row[24]), margin=float(row[25]), lobulation=float(row[26]), spiculation=float(row[27]),
                maligancy=float(row[29])
            ))
    return nodules


def count_high(list):
    cnt = 0
    for npy in list:
        if 'high' in npy:
            cnt += 1
    return cnt

def count_low(list):
    cnt = 0
    for npy in list:
        if 'high' in npy:
            cnt += 1
    return cnt


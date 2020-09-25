import mne
import numpy as np
from scipy.io import loadmat
import os
from mne.datasets import fetch_fsaverage
import re
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import cm
from tqdm import tqdm
from multiprocessing import Process, Manager
from threading import Thread
import random
from math import ceil

import sys
import time

class MNE_Repo_Mat:
    subjects_dir = '';
    subject = '';
    bem_sol = mne.bem.ConductorModel()

    def __init__(self):
        self.__st_eeg = None
        self.__band_powers = []

    def load_data_mat(self, filename):
        self.data_mat = loadmat(filename, squeeze_me=True, struct_as_record=False)
        self.behavResp = self.data_mat['behavResp']
        self.RT = self.data_mat['RT']
        self.trigs = self.data_mat['trigs']
        self.epochs_raw = self.data_mat['epochs'].transpose(2,0,1)
        self.t = self.data_mat['t']
        self.Fs = self.data_mat['Fs']
        self.NumChannels = self.data_mat['NumChannels']
        self.chanNames = self.data_mat['chanNames'].tolist()
        return self.data_mat

    @staticmethod
    def construct_subject():
        MNE_Repo_Mat.subjects_dir = os.path.dirname(fetch_fsaverage())
        MNE_Repo_Mat.subject='fsaverage'
        return MNE_Repo_Mat.subject

    @staticmethod
    def construct_montage(kind, path):
        montage = mne.channels.read_montage(kind=kind, path=path, unit='auto', transform=False)
        # montage.kind = '3d'
        # montage.plot()
        return montage

    def construct_info(self, montage = None, sfreq = 500):
        if montage is None:
            montage = MNE_Repo_Mat.construct_montage('neuroscan64ch', 'montages')
        self.info = mne.create_info(montage.ch_names, sfreq, ch_types='eeg', montage=montage)
        return self.info


    def construct_events(self, trigs):
        number_of_trials = len(trigs)
        events = np.zeros((number_of_trials, 3), dtype=int)

        for i in range(len(trigs)):
            events[i,0] = i
            events[i,2] = trigs[i]
        return events


    def construct_epoch_array(self, tmin, events = None):
        self.epochs = mne.EpochsArray(self.epochs_raw, info=self.info, tmin=tmin, events=events)
        self.epochs.apply_baseline((None, 0))
        self.event_ids = self.epochs.event_id
        return self.epochs

    def save_epochs(self, epochs):
        for key in epochs:
            epoch_path_to_save = 'combined_epochs/' + key  + '.fif'
            epochs[key].save(epoch_path_to_save, overwrite=True)

    def load_epochs(self, path):
        self.epochs = mne.read_epochs(path, verbose=0)
        return self.epochs

    # def get_trigger_wise_epochs(self, epochs, event, event_ids):
    #     trig_wise_epochs = dict()
    #     trig_wise_epochs.keys = event_ids
    #     for epoch in epochs:

    def construct_evoked_array(self, method):
        evoked = self.epochs.average(method=method)
#         evoked.apply_baseline((None, 0))
        evoked.set_eeg_reference(projection=True)
        evoked.apply_proj()
        evoked.plot(spatial_colors=True,unit=False)
        evoked.plot_topomap(times=[0.1], size=3)
        return evoked

    def construct_trigger_wise_evoked_array(self, epoch, event_ids, method):
        trig_wise_evoked = dict()
        for key in event_ids:
            evoked = epoch[key].average(method=method)
            # evoked.apply_baseline(baseline=(-0.2, 0))
            evoked.set_eeg_reference(projection=True)
            evoked.apply_proj()
            trig_wise_evoked[key] = evoked
            del evoked
        return trig_wise_evoked

    def save_trigger_wise_evokeds(self, evokeds):
        for key in evokeds:
            sub_folder_path = 'ERPs/' + key
            if not os.path.exists(sub_folder_path):
                os.mkdir(sub_folder_path)
            for event_id in evokeds[key]:
                erp_path_save = sub_folder_path + '/' + event_id + '_ave.fif'
                if os.path.exists(erp_path_save):
                    os.remove(erp_path_save)
                evokeds[key][event_id].save(erp_path_save)

    def load_trigger_wise_evokeds(self, folder_path, event_ids):
        trig_wise_evoked = dict()
        for key in event_ids:
            erp_path = folder_path + '/' + key + '_ave.fif'
            trig_wise_evoked[key] = mne.Evoked(erp_path)
        return trig_wise_evoked


    @staticmethod
    def setup_src_space():
        if not os.path.exists('source_space/src_space.fif'):
            src = mne.setup_source_space(MNE_Repo_Mat.subject, spacing='oct6')
            src.save('source_space/src_space.fif')
        else:
            src = mne.read_source_spaces('source_space/src_space.fif')
        return src

    @staticmethod
    def setup_bem():
        if not os.path.exists('bem/fsaverage_bem.fif'):
            model = mne.make_bem_model(MNE_Repo_Mat.subject)
            bem_sol = mne.make_bem_solution(model)
            mne.write_bem_solution('bem/fsaverage_bem.fif',bem_sol)
        else:
            bem_sol = mne.read_bem_solution('bem/fsaverage_bem.fif')
        return bem_sol

    @staticmethod
    def get_trans_obj():
        data_path = mne.datasets.sample.data_path()
        trans = mne.read_trans(data_path + '/MEG/sample/sample_audvis_raw-trans.fif')
        return trans

    def compute_forward_sol(self, info, src, bem):
        trans = MNE_Repo_Mat.get_trans_obj()
        fwd = mne.make_forward_solution(info=info, trans=trans, src=src, bem=bem)
        return fwd

    def compute_covariance_mat(self, epochs):
        return mne.compute_covariance(epochs, tmax=0., method=['shrunk', 'empirical'], rank=None)

    def create_inverse_operator(self, info, cov, fwd, loose, depth):
        return mne.minimum_norm.make_inverse_operator(info=info, noise_cov=cov, forward=fwd, loose=loose, depth=depth)

    def apply_inverse_operator_with_residual(self, evoked, inv, lambda2, ori, method, residual, verbose):
        stc, residual = mne.minimum_norm.apply_inverse(evoked, inv, lambda2,
                              method=method, pick_ori=ori,
                              return_residual=residual, verbose=verbose)
        return stc, residual

    def apply_inverse_operator(self, evoked, inv, lambda2, ori, method, verbose):
        stc = mne.minimum_norm.apply_inverse(evoked, inv, lambda2,
                              method=method, pick_ori=ori, verbose=verbose)
        return stc

    def apply_inverse_operator_event_wise(self, epoch, evoked, info, fwd, lambda2, ori, method, verbose):
        stc_single_sub = dict()

        for event_id in evoked:
            cov = self.compute_covariance_mat(epoch[event_id])
            inv = self.create_inverse_operator(info, cov, fwd, 0.2, 0.8)


            stc_single_sub[event_id] = self.apply_inverse_operator(evoked[event_id], inv, lambda2, ori, method, verbose)
        return stc_single_sub


    # def apply_inverse_operator_event_wise(self, evoked, inv, lambda2, ori, method, verbose):
    #     stc = mne.minimum_norm.apply_inverse(evoked, inv, lambda2,
    #                           method=method, pick_ori=ori, verbose=verbose)
    #     return stc

    @staticmethod
    def init_exp_for_sl():
        MNE_Repo_Mat.construct_subject()
        montage = MNE_Repo_Mat.construct_montage('neuroscan64ch', 'montages')
        src = MNE_Repo_Mat.setup_src_space()
        bem = MNE_Repo_Mat.setup_bem()
        return montage, src, bem

    def __construct_st_epoch_array(self, tmin=-0.2):
        st_epoch = mne.EpochsArray(self.__st_eeg, info=self.info, tmin=tmin, verbose=False)
        return st_epoch

    def get_avg_band_power(self, tmin=0, tmax=0.2):
        import itertools
        band_powers = []
        for i in range(len(self.epochs_raw)):
            self.__st_eeg = self.epochs_raw[i:i+1]

            st_epoch = self.__construct_st_epoch_array(-0.2)

            psd_alpha, _ = mne.time_frequency.psd_welch(st_epoch, 8, 15, tmin=tmin, tmax=tmax, n_fft=self.Fs, n_per_seg=self.Fs, verbose=False)
            psd_beta, _ = mne.time_frequency.psd_welch(st_epoch, 16, 31, tmin=tmin, tmax=tmax, n_fft=self.Fs, n_per_seg=self.Fs, verbose=False)
            psd_gamma, _ = mne.time_frequency.psd_welch(st_epoch, 32, 60, tmin=tmin, tmax=tmax, n_fft=self.Fs, n_per_seg=self.Fs, verbose=False)

            band_pow_alpha = np.average(psd_alpha, axis=2).flatten()
            band_pow_beta = np.average(psd_beta, axis=2).flatten()
            band_pow_gamma = np.average(psd_gamma, axis=2).flatten()

            band_pows_st = [band_pow_alpha, band_pow_beta, band_pow_gamma]
            band_pows_st = list(itertools.chain(*band_pows_st))

            band_powers.append(band_pows_st)

        return np.array(band_powers)

    def plot_combine_topomaps(self, start, end, subject):
        folder_path = 'band_power_topomap_BTS_figs/'
        subject_path = folder_path + subject
        alpha_path = subject_path + '/alpha'
        beta_path = subject_path + '/beta'
        gamma_path = subject_path + '/gamma'
        combined_path = subject_path + '/combined'

        if not os.path.exists(combined_path):
            os.mkdir(combined_path)

        for i in range(start, end):
            img_path_alpha = alpha_path +  '/bts_' + str(i+1) + '.png'
            img_path_beta = beta_path  +  '/bts_' + str(i+1) + '.png'
            img_path_gamma = gamma_path +  '/bts_' + str(i+1) + '.png'

            alpha = cv2.imread(img_path_alpha, cv2.IMREAD_GRAYSCALE)
            beta = cv2.imread(img_path_beta, cv2.IMREAD_GRAYSCALE)
            gamma = cv2.imread(img_path_gamma, cv2.IMREAD_GRAYSCALE)

            c_img = np.dstack((alpha, beta, gamma))

            img_path = combined_path + '/trial_' + str(i+1) + '.png'

            cv2.imwrite(img_path, c_img)

    def plot_topomap_avg_bp(self, start, end, subject, avg_power_st_slice, fig_dict):
        folder_path = 'band_power_topomap_BTS_200/'
        subject_path = folder_path + subject
        alpha_path = subject_path + '/alpha'
        beta_path = subject_path + '/beta'
        gamma_path = subject_path + '/gamma'
        combined_path = subject_path + '/combined'

        if not os.path.exists(subject_path):
            os.mkdir(subject_path)
        if not os.path.exists(alpha_path):
            os.mkdir(alpha_path)
        if not os.path.exists(beta_path):
            os.mkdir(beta_path)
        if not os.path.exists(gamma_path):
            os.mkdir(gamma_path)

        for i, trial in zip(range(start, end), avg_power_st_slice):
            alpha = trial[0:64]
            beta = trial[64:128]
            gamma = trial[128:192]

            img_path_alpha = alpha_path +  '/bts_' + str(i+1) + '.pkl'
            img_path_beta = beta_path  +  '/bts_' + str(i+1) + '.pkl'
            img_path_gamma = gamma_path +  '/bts_' + str(i+1) + '.pkl'

            # topo_alpha, _ = mne.viz.plot_topomap(alpha, self.info, res=256, show=False, contours=0, cmap=cm.gray_r)
            # topo_alpha.get_figure().savefig(img_path_alpha, dpi=64)
            #
            # topo_beta, _ = mne.viz.plot_topomap(beta, self.info, res=256, show=False, contours=0, cmap=cm.gray_r)
            # topo_beta.get_figure().savefig(img_path_beta, dpi=64)
            #
            # topo_gamma, _ = mne.viz.plot_topomap(gamma, self.info, res=256, show=False, contours=0, cmap=cm.gray_r)
            # topo_gamma.get_figure().savefig(img_path_gamma, dpi=64)

            evoked_alpha = mne.EvokedArray(alpha.reshape(len(alpha), 1), self.info)
            evoked_beta = mne.EvokedArray(beta.reshape(len(alpha), 1), self.info)
            evoked_gamma = mne.EvokedArray(gamma.reshape(len(alpha), 1), self.info)



            topo_alpha = evoked_alpha.plot_topomap(times = [0], show=False, contours=0, size=5,
                                            colorbar=False, title=None)
            topo_beta = evoked_beta.plot_topomap(times = [0], show=False, contours=0, size=5,
                                            colorbar=False, title=None)
            topo_gamma = evoked_gamma.plot_topomap(times = [0], show=False, contours=0, size=5,
                                            colorbar=False, title=None)

            pickle.dump(topo_alpha, open(img_path_alpha, "wb"))
            pickle.dump(topo_beta, open(img_path_beta, "wb"))
            pickle.dump(topo_gamma, open(img_path_gamma, "wb"))

            # fig_dict['alpha'].append((topo_alpha, img_path_alpha))
            # fig_dict['beta'].append((topo_beta, img_path_beta))
            # fig_dict['gamma'].append((topo_gamma, img_path_gamma))
            # time.sleep(0.1)
            #
            # print("saved")
            # sys.stdout.flush()


    def save_topomap(self, fig_tuples):

        for fig, path in fig_tuples:
            fig.savefig(path)



    def async_save_band_power_topo(self, subject, avg_power_st):
        n = list(range(0, len(avg_power_st), 50))
        n.append(len(avg_power_st))

        manager = Manager()

        fig_dict = manager.dict()

        fig_dict['alpha'] = manager.list()
        fig_dict['beta'] = manager.list()
        fig_dict['gamma'] = manager.list()
        processes = []
        for i in range(len(n) - 1):
            process = Process(target=self.plot_topomap_avg_bp, args=(n[i], n[i+1], subject, avg_power_st[n[i]:n[i+1]], fig_dict), name='Process s_{} n_{}'.format(subject, i))
            processes.append(process)
            process.start()
            # elf.plot_topomap_avg_bp(n[i], n[i+1], subject, avg_power_st[n[i]:n[i+1]])

        for process in processes:
            process.join()

        # p_alpha = Process(target=self.save_topomap, args=(fig_dict['alpha']))

        # self.save_topomap(fig_dict['alpha'])
        # self.save_topomap(fig_dict['beta'])
        # self.save_topomap(fig_dict['gamma'])


    def async_save_combined_topomap(self, subject, no_of_trials):
        n = list(range(0, no_of_trials, 50))
        n.append(no_of_trials)

        processes = []

        for i in range(len(n) - 1):
            process = Process(target=self.plot_combine_topomaps, args=(n[i], n[i+1], subject), name='Process c_{} n_{}'.format(subject, i))
            processes.append(process)
            process.start()
            # self.plot_combine_topomaps(n[i], n[i+1], subject)

        for process in processes:
            process.join()



    def train_test_spliter_ML(self, subjects, label_mappers, labels = [1,2,3,4,5], folder_path='band_power_topomap_new', data_path='combined', save_path='data'):

        def save(subject, img_names, labels, dir, save_dir):
            for img_name, lbl in zip(img_names, labels):
                load_path = dir + '/' + img_name
                img = cv2.imread(load_path)
                save_path = save_dir + '/' + str(lbl) + '/' + subject + '_' + img_name
                cv2.imwrite(save_path, img)

        def get_indexes_val_test(indexes, n_trials, sub_labels, labels, split_ratio=0.2):
            indices = []
            split_ratio = split_ratio / len(labels)
            for lbl in labels:
                lbl_indices = [i for i in indexes if sub_labels[i] == lbl]
                r_index = random.sample(lbl_indices, ceil(n_trials * split_ratio))
                indices.extend(r_index)
            return indices


        train_path = save_path + '/' + 'train'
        test_path = save_path + '/' + 'test'
        validation_path = save_path + '/' + 'validation'

        if not os.path.exists(save_path):
            os.mkdir(save_path)
            os.mkdir(train_path)
            os.mkdir(test_path)
            os.mkdir(validation_path)
            for label in labels:
                os.mkdir(train_path + '/' + str(label))
                os.mkdir(test_path + '/' + str(label))
                os.mkdir(validation_path + '/' + str(label))

        if data_path is None:
            return 'Cannot do with data_path None'

        for subject in tqdm(subjects):
            d_path = folder_path + '/' + subject + '/' + data_path
            trials = np.array(os.listdir(d_path))
            n_trials = len(trials)

            trial_indexes = list(range(0, n_trials))

            test_indexes = get_indexes_val_test(trial_indexes, n_trials, label_mappers[subject], labels)
            print(len(test_indexes))
            temp = np.delete(trial_indexes, test_indexes).tolist()
            validation_indexes = get_indexes_val_test(temp, n_trials, label_mappers[subject], labels, split_ratio=0.1)
            print(len(validation_indexes))
            test_imgs = trials[test_indexes]
            validation_imgs = trials[validation_indexes]

            test_lbls = label_mappers[subject][test_indexes]
            validation_lbls = label_mappers[subject][validation_indexes]

            test_val_indexes = test_indexes
            test_val_indexes.extend(validation_indexes)
            training_imgs = np.delete(trials, test_val_indexes)
            training_lbls = np.delete(label_mappers[subject], test_val_indexes)

            tr_process = Process(target=save, args=(subject, training_imgs, training_lbls, d_path, train_path), name='process training {}'.format(subject))
            ts_process = Process(target=save, args=(subject, test_imgs, test_lbls, d_path, test_path), name='process test {}'.format(subject))
            va_process = Process(target=save, args=(subject, validation_imgs, validation_lbls, d_path, validation_path), name='process validation {}'.format(subject))

            tr_process.start()
            ts_process.start()
            va_process.start()

            tr_process.join()
            ts_process.join()
            va_process.join()




    def generate_source_estimate_straight(self, file_name, montage, src, bem):
        self.load_data(file_name)

        info = self.construct_info(montage)

        epochs = self.construct_epoch_array(-0.2)
        evoked = self.construct_evoked_array('mean')

        fwd = self.compute_forward_sol(info, src, bem)
        cov = self.compute_covariance_mat(epochs)
        inv = self.create_inverse_operator(info, cov, fwd, 0.2, 0.8)

        snr = 3.
        lambda2 = 1. / snr ** 2
        stc,residual = self.apply_inverse_operator_with_residual(evoked, inv, lambda2,  None, 'sLORETA', True, True)

        return stc,residual

    def save_event_wise_source_estimates(self, stcs):
        for stc_sub in stcs:
            stc_sub_path = 'stcs/' + stc_sub
            if not os.path.exists(stc_sub_path):
                os.mkdir(stc_sub_path)
            for event_id in stcs[stc_sub]:
                event_stc_path = stc_sub_path + '/' + event_id
                stcs[stc_sub][event_id].save(fname = event_stc_path, ftype = 'stc')

    def load_stcs(self, path, subject):
        stc_sub_path = path + '/' + subject + '/'
        event_stcs = dict()
        for event_stc_file in os.listdir(stc_sub_path):
            if event_stc_file.endswith('.stc') and 'lh' in event_stc_file:
                event_key = re.findall(r'\d+', event_stc_file)[0]
                event_stc_path = stc_sub_path + event_stc_file
                event_stcs[event_key] = mne.read_source_estimate(event_stc_path)
        return event_stcs


    def generate_ERPs(self, filenames, montage, gen_mode = True, save = True):
        self.evokeds = dict()
        self.epochs_dict = dict()

        for filename in filenames:
            self.load_data(filename)
            info = self.construct_info(montage)
            erp_subject_key = re.split(r'[./]', filename)[1] + '_erp'
            epoch_subject_key = re.split(r'[./]', filename)[1] + '_epoch'

            events = self.construct_events(self.trigs)
            m_events = mne.merge_events(events, [1,5], 15)

            if gen_mode:
                self.epochs_dict[epoch_subject_key]  = self.construct_epoch_array(-0.2, m_events)
                self.evokeds[erp_subject_key] = self.construct_trigger_wise_evoked_array(self.epochs_dict[epoch_subject_key], self.epochs_dict[epoch_subject_key].event_id, 'mean')
            else:
                epoch_path = 'epochs/' + epoch_subject_key + '.fif'
                erp_folder_path = 'ERPs/' + erp_subject_key
                self.epochs_dict[epoch_subject_key] = self.load_epochs(epoch_path)
                self.evokeds[erp_subject_key] = self.load_trigger_wise_evokeds(erp_folder_path, self.epochs_dict[epoch_subject_key].event_id)

        if save:
            self.save_trigger_wise_evokeds(self.evokeds)
            self.save_epochs(self.epochs_dict)

        return self.evokeds, self.epochs_dict

    def generate_event_wise_stcs(self, epochs, evokeds, montage, src, bem, gen_mode = True, save = True):

        info = self.construct_info(montage)

        self.stcs = dict()

        for sub_key_epoch, sub_key_erp in zip(epochs, evokeds):

            sub_stc_key = re.split(r'[_]', sub_key_erp)[0] + '_stc'

            if not gen_mode:
                self.stcs[sub_stc_key] = self.load_stcs('stcs' ,sub_stc_key)
                continue

            fwd = self.compute_forward_sol(info, src, bem)

            snr = 3.
            lambda2 = 1. / snr ** 2

            self.stcs[sub_stc_key] = self.apply_inverse_operator_event_wise(epochs[sub_key_epoch], evokeds[sub_key_erp], info, fwd, lambda2, None, 'sLORETA', None)


        if save:
            self.save_event_wise_source_estimates(self.stcs)

        return self.stcs

    def apply_cortical_parcellation_event_stcs(self, stcs, src, save=True, gen_mode=True):

        labels = mne.read_labels_from_annot(self.subject)
        self.labels = [lbl for lbl in labels if lbl.name != 'unknown-lh']
        stc_path = 'stcs/'
        self.stc_cp = dict()

        for key, event_stcs in stcs.items():
            stc_sub_path = stc_path + key + '/'
            event_stcs_cp = np.zeros((68, 500, 5))
            for event_id, event_stc in event_stcs.items():
                event_stc_path = stc_sub_path + event_id + '.csv'
                if gen_mode:
                    label_tc = mne.extract_label_time_course(event_stc, self.labels, src, mode='pca_flip')
                else:
                    label_tc = np.genfromtxt(event_stc_path, delimiter = ',')
                event_stcs_cp[:, :, int(event_id)-1] = label_tc
                if save:
                    np.savetxt(event_stc_path, label_tc, delimiter = ',')
            self.stc_cp[key] = event_stcs_cp
        return self.stc_cp

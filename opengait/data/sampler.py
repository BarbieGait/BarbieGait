import math
import random
import torch
import torch.distributed as dist
import torch.utils.data as tordata
from remote_pdb import set_trace
import random

class DualSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size, batch_shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        if len(self.batch_size) != 2:
            raise ValueError(
                "batch_size should be (P x K) not {}".format(batch_size))
        self.batch_shuffle = batch_shuffle

        self.world_size = dist.get_world_size()
        if (self.batch_size[0]*self.batch_size[1]) % self.world_size != 0:
            raise ValueError("World size ({}) is not divisible by batch_size ({} x {})".format(
                self.world_size, batch_size[0], batch_size[1]))
        self.rank = dist.get_rank()

    def __iter__(self):
        while True:
            sample_indices = []
            pid_list = sync_random_sample_list(
                self.dataset.label_set, self.batch_size[0])
            view = random.choice(['Camera0', 'Camera1', 'Camera10', 'Camera11', 'Camera12', 'Camera13', 'Camera14', \
                                  'Camera15', 'Camera2', 'Camera3', 'Camera4', 'Camera5', 'Camera6', 'Camera7', 'Camera8', 'Camera9'])

            for pid in pid_list:
                indices = self.dataset.indices_dict[pid]

                indices_types_1 = self.dataset.indices_thk_dict['thick2'] + self.dataset.indices_thk_dict['thick3'] + self.dataset.indices_thk_dict['thick4']
                indices_types_2 = self.dataset.indices_thk_dict['thick5'] + self.dataset.indices_thk_dict['thick6'] + self.dataset.indices_thk_dict['thick7'] \
                                + self.dataset.indices_thk_dict['thick8'] + self.dataset.indices_thk_dict['thick9'] + self.dataset.indices_thk_dict['thick10']

                indices_view = self.dataset.indices_view_dict[view]
                intersection_1 = list(set(indices) & set(indices_types_1))
                intersection_2 = list(set(indices) & set(indices_types_2))
                intersection_3 = list(set(indices) & set(indices_view))

                intersection_1 = list(set(intersection_1) & set(intersection_3))
                intersection_2 = list(set(intersection_2) & set(intersection_3))

                indices_1 = sync_random_sample_list(
                    intersection_1, k=(self.batch_size[1]) // 2)

                indices_2 = sync_random_sample_list(
                    intersection_2, k=(self.batch_size[1]) // 2)

                final_indices = indices_1 + indices_2
                # indices = sync_random_sample_list(
                #     indices, k=self.batch_size[1])

                sample_indices += final_indices

            if self.batch_shuffle:
                sample_indices = sync_random_sample_list(
                    sample_indices, len(sample_indices))

            total_batch_size = self.batch_size[0] * self.batch_size[1]
            total_size = int(math.ceil(total_batch_size /
                                       self.world_size)) * self.world_size
            sample_indices += sample_indices[:(
                total_batch_size - len(sample_indices))]

            sample_indices = sample_indices[self.rank:total_size:self.world_size]
            yield sample_indices

    def __len__(self):
        return len(self.dataset)

class DualSampler2(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size, batch_shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        if len(self.batch_size) != 2:
            raise ValueError(
                "batch_size should be (P x K) not {}".format(batch_size))
        self.batch_shuffle = batch_shuffle

        self.world_size = dist.get_world_size()
        if (self.batch_size[0]*self.batch_size[1]) % self.world_size != 0:
            raise ValueError("World size ({}) is not divisible by batch_size ({} x {})".format(
                self.world_size, batch_size[0], batch_size[1]))
        self.rank = dist.get_rank()

    def __iter__(self):
        while True:
            sample_indices = []
            pid_list = sync_random_sample_list(
                self.dataset.label_set, self.batch_size[0])
            view = random.choice(['Camera0', 'Camera1', 'Camera10', 'Camera11', 'Camera12', 'Camera13', 'Camera14', \
                                  'Camera15', 'Camera2', 'Camera3', 'Camera4', 'Camera5', 'Camera6', 'Camera7', 'Camera8', 'Camera9'])

            for pid in pid_list:
                indices = self.dataset.indices_dict[pid]

                indices_types_1 = self.dataset.indices_thk_dict['thick2'] + self.dataset.indices_thk_dict['thick3'] + self.dataset.indices_thk_dict['thick4']
                indices_types_2 = self.dataset.indices_thk_dict['thick5'] + self.dataset.indices_thk_dict['thick6'] + self.dataset.indices_thk_dict['thick7'] \
                                + self.dataset.indices_thk_dict['thick8'] + self.dataset.indices_thk_dict['thick9'] + self.dataset.indices_thk_dict['thick10']

                indices_view = self.dataset.indices_view_dict[view]
                intersection_1 = list(set(indices) & set(indices_types_1))
                intersection_2 = list(set(indices) & set(indices_types_2))
                intersection_3 = list(set(indices) & set(indices_view))

                intersection_1 = list(set(intersection_1) & set(intersection_3))
                intersection_2 = list(set(intersection_2) & set(intersection_3))

                indices_1 = sync_random_sample_list(
                    intersection_1, k=(self.batch_size[1]) // 2)

                indices_2 = sync_random_sample_list(
                    intersection_2, k=(self.batch_size[1]) // 2)

                final_indices = indices_1 + indices_2
                # indices = sync_random_sample_list(
                #     indices, k=self.batch_size[1])

                sample_indices += final_indices

            if self.batch_shuffle:
                sample_indices = sync_random_sample_list(
                    sample_indices, len(sample_indices))

            total_batch_size = self.batch_size[0] * self.batch_size[1]
            total_size = int(math.ceil(total_batch_size /
                                       self.world_size)) * self.world_size
            sample_indices += sample_indices[:(
                total_batch_size - len(sample_indices))]

            sample_indices = sample_indices[self.rank:total_size:self.world_size]
            yield sample_indices

    def __len__(self):
        return len(self.dataset)

class TripletSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size, batch_shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        if len(self.batch_size) != 2:
            raise ValueError(
                "batch_size should be (P x K) not {}".format(batch_size))
        self.batch_shuffle = batch_shuffle

        self.world_size = dist.get_world_size()
        if (self.batch_size[0]*self.batch_size[1]) % self.world_size != 0:
            raise ValueError("World size ({}) is not divisible by batch_size ({} x {})".format(
                self.world_size, batch_size[0], batch_size[1]))
        self.rank = dist.get_rank()

    def __iter__(self):
        while True:
            sample_indices = []
            pid_list = sync_random_sample_list(
                self.dataset.label_set, self.batch_size[0])

            for pid in pid_list:
                indices = self.dataset.indices_dict[pid]
                indices = sync_random_sample_list(
                    indices, k=self.batch_size[1])
                sample_indices += indices

            if self.batch_shuffle:
                sample_indices = sync_random_sample_list(
                    sample_indices, len(sample_indices))

            total_batch_size = self.batch_size[0] * self.batch_size[1]
            total_size = int(math.ceil(total_batch_size /
                                       self.world_size)) * self.world_size
            sample_indices += sample_indices[:(
                total_batch_size - len(sample_indices))]

            sample_indices = sample_indices[self.rank:total_size:self.world_size]
            yield sample_indices

    def __len__(self):
        return len(self.dataset)

class TripletSampler_DA(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size, batch_shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        if len(self.batch_size) != 2:
            raise ValueError(
                "batch_size should be (P x K) not {}".format(batch_size))
        self.batch_shuffle = batch_shuffle

        self.world_size = dist.get_world_size()
        if (self.batch_size[0]*self.batch_size[1]) % self.world_size != 0:
            raise ValueError("World size ({}) is not divisible by batch_size ({} x {})".format(
                self.world_size, batch_size[0], batch_size[1]))
        self.rank = dist.get_rank()

    def __iter__(self):
        while True:
            sample_indices = []
            pid_list = sync_random_sample_list(
                self.dataset.label_set, self.batch_size[0])

            for pid in pid_list:
                indices = self.dataset.indices_dict[pid]
                indices = sync_random_sample_list(
                    indices, k=self.batch_size[1])
                sample_indices += indices

            if self.batch_shuffle:
                sample_indices = sync_random_sample_list(
                    sample_indices, len(sample_indices))

            total_batch_size = self.batch_size[0] * self.batch_size[1]
            total_size = int(math.ceil(total_batch_size /
                                       self.world_size)) * self.world_size
            sample_indices += sample_indices[:(
                total_batch_size - len(sample_indices))]

            sample_indices = sample_indices[self.rank:total_size:self.world_size]
            yield sample_indices

    def __len__(self):
        return len(self.dataset)

def sortbynum(indice):
    return(int(indice))
class AllthkSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size, batch_shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        if len(self.batch_size) != 2:
            raise ValueError(
                "batch_size should be (P x K) not {}".format(batch_size))
        self.batch_shuffle = batch_shuffle

        self.world_size = dist.get_world_size()
        if (self.batch_size[0]*self.batch_size[1]) % self.world_size != 0:
            raise ValueError("World size ({}) is not divisible by batch_size ({} x {})".format(
                self.world_size, batch_size[0], batch_size[1]))
        self.rank = dist.get_rank()

    def __iter__(self):
        while True:
            sample_indices = []
            # pid_list = sync_random_sample_list(
            #     self.dataset.label_set, self.batch_size[0])
            # view = random.choice(['Camera0', 'Camera1', 'Camera10', 'Camera11', 'Camera12', 'Camera13', 'Camera14', \
            #                       'Camera15', 'Camera2', 'Camera3', 'Camera4', 'Camera5', 'Camera6', 'Camera7', 'Camera8', 'Camera9'])
            view = 'Camera1'
            pid_list = ['538','538','538','538']
            # pid_list = ['357','357','357','357']
            for pid in pid_list:
                indices = self.dataset.indices_dict[pid]

                indices_thk0 = self.dataset.indices_thk_dict['thick0']
                indices_thk1 = self.dataset.indices_thk_dict['thick0']
                indices_thk2 = self.dataset.indices_thk_dict['thick0']
                indices_thk3 = self.dataset.indices_thk_dict['thick0']
                indices_thk4 = self.dataset.indices_thk_dict['thick0']
                indices_thk6 = self.dataset.indices_thk_dict['thick0']
                indices_clo0 = self.dataset.indices_clo_dict['12']
                indices_clo1 = self.dataset.indices_clo_dict['14']
                indices_clo2 = self.dataset.indices_clo_dict['15']
                indices_clo3 = self.dataset.indices_clo_dict['7']
                indices_clo4 = self.dataset.indices_clo_dict['9']
                # indices_clo5 = self.dataset.indices_clo_dict['72']
                indices_clo6 = self.dataset.indices_clo_dict['10']

                # indices_thk0 = self.dataset.indices_thk_dict['thick0']
                # indices_thk1 = self.dataset.indices_thk_dict['thick1']
                # indices_thk2 = self.dataset.indices_thk_dict['thick2']
                # indices_thk3 = self.dataset.indices_thk_dict['thick3']
                # indices_thk4 = self.dataset.indices_thk_dict['thick4']
                # # indices_thk5 = self.dataset.indices_thk_dict['thick5']
                # indices_thk6 = self.dataset.indices_thk_dict['thick6']
                # # indices_thk7 = self.dataset.indices_thk_dict['thick7']
                # # indices_thk8 = self.dataset.indices_thk_dict['thick8']
                # # indices_thk9 = self.dataset.indices_thk_dict['thick9']

                # indices_clo0 = self.dataset.indices_clo_dict['1']
                # indices_clo1 = self.dataset.indices_clo_dict['0']
                # indices_clo2 = self.dataset.indices_clo_dict['5']
                # indices_clo3 = self.dataset.indices_clo_dict['6']
                # indices_clo4 = self.dataset.indices_clo_dict['11']
                # # indices_clo5 = self.dataset.indices_clo_dict['72']
                # indices_clo6 = self.dataset.indices_clo_dict['32']

                indices_view = self.dataset.indices_view_dict[view]
                intersection_0 = list(set(indices) & set(indices_thk0) & set(indices_view) & set(indices_clo0))
                intersection_1 = list(set(indices) & set(indices_thk1) & set(indices_view) & set(indices_clo1))
                intersection_2 = list(set(indices) & set(indices_thk2) & set(indices_view) & set(indices_clo2))
                intersection_3 = list(set(indices) & set(indices_thk3) & set(indices_view) & set(indices_clo3))
                intersection_4 = list(set(indices) & set(indices_thk4) & set(indices_view) & set(indices_clo4))
                # intersection_5 = list(set(indices) & set(indices_thk5) & set(indices_view) & set(indices_clo5))
                intersection_6 = list(set(indices) & set(indices_thk6) & set(indices_view) & set(indices_clo6))
                # intersection_7 = list(set(indices) & set(indices_thk7) & set(indices_view))
                # intersection_8 = list(set(indices) & set(indices_thk8) & set(indices_view))
                # intersection_9 = list(set(indices) & set(indices_thk9) & set(indices_view))

                indices_0 = fix_sample_list(intersection_0, k=1)
                indices_1 = fix_sample_list(intersection_1, k=1)
                indices_2 = fix_sample_list(intersection_2, k=1)
                indices_3 = fix_sample_list(intersection_3, k=1)
                indices_4 = fix_sample_list(intersection_4, k=1)
                # indices_5 = fix_sample_list(intersection_5, k=1)
                indices_6 = fix_sample_list(intersection_6, k=1)

                # indices_7 = sync_random_sample_list(intersection_7, k=(self.batch_size[1])//10)
                # indices_8 = sync_random_sample_list(intersection_8, k=(self.batch_size[1])//10)
                # indices_9 = sync_random_sample_list(intersection_9, k=(self.batch_size[1])//10)

                final_indices = indices_0 + indices_1 + indices_2 + indices_3 + indices_4 + indices_6
                # final_indices = sorted(final_indices,key=sortbynum)

                sample_indices += final_indices

            if self.batch_shuffle:
                sample_indices = sync_random_sample_list(
                    sample_indices, len(sample_indices))

            total_batch_size = self.batch_size[0] * self.batch_size[1]
            total_size = int(math.ceil(total_batch_size /
                                       self.world_size)) * self.world_size
            sample_indices += sample_indices[:(
                total_batch_size - len(sample_indices))]

            sample_indices = sample_indices[self.rank:total_size:self.world_size]
            yield sample_indices

def sync_random_sample_list(obj_list, k, common_choice=False):
    if common_choice:
        idx = random.choices(range(len(obj_list)), k=k)
        idx = torch.tensor(idx)
    if len(obj_list) < k:
        idx = random.choices(range(len(obj_list)), k=k)
        idx = torch.tensor(idx)
    else:
        idx = torch.randperm(len(obj_list))[:k]
    if torch.cuda.is_available():
        idx = idx.cuda()
    torch.distributed.broadcast(idx, src=0)
    idx = idx.tolist()

    return [obj_list[i] for i in idx]

def fix_sample_list(obj_list, k, common_choice=False):
    if common_choice:
        idx = random.choices(range(len(obj_list)), k=k)
        idx = torch.tensor(idx)
    if len(obj_list) < k:
        idx = random.choices(range(len(obj_list)), k=k)
        idx = torch.tensor(idx)
    else:
        idx = range(len(obj_list))[:k]
        idx = torch.tensor(idx)
    if torch.cuda.is_available():
        idx = idx.cuda()
    torch.distributed.broadcast(idx, src=0)
    idx = idx.tolist()
    return [obj_list[i] for i in idx]


class InferenceSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        self.size = len(dataset)
        indices = list(range(self.size))

        world_size = dist.get_world_size()
        rank = dist.get_rank()

        if batch_size % world_size != 0:
            raise ValueError("World size ({}) is not divisible by batch_size ({})".format(
                world_size, batch_size))

        if batch_size != 1:
            complement_size = math.ceil(self.size / batch_size) * \
                batch_size
            indices += indices[:(complement_size - self.size)]
            self.size = complement_size

        batch_size_per_rank = int(self.batch_size / world_size)
        indx_batch_per_rank = []

        for i in range(int(self.size / batch_size_per_rank)):
            indx_batch_per_rank.append(
                indices[i*batch_size_per_rank:(i+1)*batch_size_per_rank])

        self.idx_batch_this_rank = indx_batch_per_rank[rank::world_size]

    def __iter__(self):
        yield from self.idx_batch_this_rank

    def __len__(self):
        return len(self.dataset)


class CommonSampler(tordata.sampler.Sampler):
    def __init__(self,dataset,batch_size,batch_shuffle):

        self.dataset = dataset
        self.size = len(dataset)
        self.batch_size = batch_size
        if isinstance(self.batch_size,int)==False:
            raise ValueError(
                "batch_size shoude be (B) not {}".format(batch_size))
        self.batch_shuffle = batch_shuffle

        self.world_size = dist.get_world_size()
        if self.batch_size % self.world_size !=0:
            raise ValueError("World size ({}) is not divisble by batch_size ({})".format(
                self.world_size, batch_size))
        self.rank = dist.get_rank()

    def __iter__(self):
        while True:
            indices_list = list(range(self.size))
            sample_indices = sync_random_sample_list(
                    indices_list, self.batch_size, common_choice=True)
            total_batch_size =  self.batch_size
            total_size = int(math.ceil(total_batch_size /
                                       self.world_size)) * self.world_size
            sample_indices += sample_indices[:(
                total_batch_size - len(sample_indices))]
            sample_indices = sample_indices[self.rank:total_size:self.world_size]
            yield sample_indices

    def __len__(self):
        return len(self.dataset)

# **************** For GaitSSB ****************
# Fan, et al: Learning Gait Representation from Massive Unlabelled Walking Videos: A Benchmark, T-PAMI2023
import random
class BilateralSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size, batch_shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.batch_shuffle = batch_shuffle

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()

        self.dataset_length = len(self.dataset)
        self.total_indices = list(range(self.dataset_length))

    def __iter__(self):
        random.shuffle(self.total_indices)
        count = 0
        batch_size = self.batch_size[0] * self.batch_size[1]
        while True:
            if (count + 1) * batch_size >= self.dataset_length:
                count = 0
                random.shuffle(self.total_indices)

            sampled_indices = self.total_indices[count*batch_size:(count+1)*batch_size]
            sampled_indices = sync_random_sample_list(sampled_indices, len(sampled_indices))

            total_size = int(math.ceil(batch_size / self.world_size)) * self.world_size
            sampled_indices += sampled_indices[:(batch_size - len(sampled_indices))]

            sampled_indices = sampled_indices[self.rank:total_size:self.world_size]
            count += 1

            yield sampled_indices * 2

    def __len__(self):
        return len(self.dataset)
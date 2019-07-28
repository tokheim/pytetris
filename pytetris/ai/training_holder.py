import random

class TrainingHolder(object):
    def __init__(self, reinforce_ratio, num_quarantine, num_reinforce, batch_size, train_ratio):
        self.reinforce_ratio = reinforce_ratio
        self.quarantine = FixedSizeList(num_quarantine)
        self.reinforcement_examples = FixedSizeList(num_reinforce)
        self.batch_size = batch_size
        self.train_ratio = train_ratio

        self.num_added = 0
        self.batches_trained = 0


    def add_examples(self, training_examples):
        num_select = int(len(training_examples) * self.reinforce_ratio)
        selected = random.sample(training_examples, num_select)
        self.quarantine.elements += selected

        unquarantined = self.quarantine.pop_overflow()
        self.reinforcement_examples.elements += unquarantined
        self.reinforcement_examples.pop_overflow()
        self.num_added += len(unquarantined)

    def batches_available(self):
        trainable = min(self.train_ratio * self.num_added, len(self.reinforcement_examples.elements))
        return int(trainable / self.batch_size)

    def training_batches(self, to_train=None):
        if to_train is None:
            to_train = self.batches_available()
        batches = []
        self.num_added -= int(to_train * self.batch_size / self.train_ratio)
        for i in range(to_train):
            batch = random.sample(self.reinforcement_examples.elements, self.batch_size)
            batches.append(batch)
            self.batches_trained += 1
        return batches

class SkewedTrainingHolder(object):
    def __init__(self, training_holders):
        self.training_holders = training_holders
        self.batches_trained = 0

    def batches_available(self):
        return min(th.batches_available() for th in self.training_holders)

    def training_batches(self, to_train=None):
        if to_train is None:
            to_train = self.batches_available()
        batchlists = [th.training_batches(to_train=to_train) for th in self.training_holders]
        batches = []
        for batch_parts in zip(*batchlists):
            batch = []
            [batch.extend(b) for b in batch_parts]
            batches.append(batch)
        self.batches_trained += to_train
        return batches

    def add_examples(self, training_examples):
        for th in self.training_holders:
            th.add_examples(training_examples)

class ConditionalTrainingHolder(object):
    def __init__(self, training_holder, condition):
        self.training_holder = training_holder
        self.condition = condition

    def add_examples(self, training_examples):
        training_examples = [te for te in training_examples if self.condition(te)]
        self.training_holder.add_examples(training_examples)

    def batches_available(self):
        return self.training_holder.batches_available()

    def training_batches(self, to_train=None):
        return self.training_holder.training_batches(to_train=to_train)

class FixedSizeList(object):
    def __init__(self, max_size):
        self.max_size = max_size
        self.elements = []

    def pop_overflow(self):
        to_remove = len(self.elements) - self.max_size
        if to_remove > 0:
            cut = self.elements[:to_remove]
            self.elements = self.elements[to_remove:]
            return cut
        return []


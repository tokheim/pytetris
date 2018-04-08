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

    def training_batches(self):
        trainable = min(self.train_ratio * self.num_added, len(self.reinforcement_examples.elements))
        to_train = int(trainable / self.batch_size)
        batches = []
        self.num_added -= int(to_train * self.batch_size / self.train_ratio)
        for i in range(to_train):
            batch = random.sample(self.reinforcement_examples.elements, self.batch_size)
            batches.append(batch)
            self.batches_trained += 1
        return batches


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


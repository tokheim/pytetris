from pytetris.gameengine import Move

class RedundantScrubber(object):
    def __init__(self):
        pass

    def scrub(self, train_examples):
        grouped = self.block_group(train_examples)
        kept = []
        for group in grouped:
            group = self.clean_dx(group)
            group = self.clean_dz(group)
            group = self.clean_premature_dy(group)
            kept += group
        return kept

    def block_group(self, train_examples):
        cur_block = -1
        groups = []
        current_group = []
        for te in train_examples:
            if te.block_num != cur_block:
                current_group = []
                groups.append(current_group)
                cur_block = te.block_num
            current_group.append(te)
        return groups

    def clean_dx(self, train_examples):
        tot_dx = 0
        kept = []
        for te in reversed(train_examples):
            dx = Move.dx(te.move)
            if dx * tot_dx >= 0:
                kept.append(te)
            tot_dx += dx
        kept.reverse()
        return kept

    def clean_dz(self, train_examples):
        tot_dz = 0
        kept = []
        for te in reversed(train_examples):
            dz = Move.dz(te.move)
            if dz == 0 or self.constructive_rotation(tot_dz, dz):
                kept.append(te)
            tot_dz += dz
        kept.reverse()
        return kept

    def constructive_rotation(self, tot_dz, dz):
        tot_dz = tot_dz % 4
        if dz < 0 and tot_dz in (0, 3):
            return True
        if dz > 0 and tot_dz in (0, 1):
            return True
        return False

    def clean_premature_dy(self, train_examples):
        stopped_move = True
        kept = []
        for te in reversed(train_examples):
            stopped_move = (stopped_move and Move.dx(te.move) == 0)
            if te.move != Move.NOTHING or stopped_move:
                kept.append(te)
        kept.reverse()
        return kept

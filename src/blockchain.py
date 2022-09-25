import numpy as np


class Block:

    def  __init__(self, height, miner, timestamp, last_miner, last_block=None):
        self.height = height
        self.miner = miner
        self.timestamp = timestamp
        self.last_miner = last_miner
        self.last_block = last_block

    def link(self, block):
        assert(self.last_miner == block.miner)      # Match last winner for link
        assert(self.height == block.height + 1)
        self.last_block = block


class BlockChain:

    def  __init__(self, num_node):
        self.num_node = num_node
        self.genesis_block = Block(0, -1, 0, -1)
        self.current_tips = [self.genesis_block]    # normal case: 1 tip; fork: >1 tips
        self.forks = np.zeros(num_node)
        self.fork_count = 0

    def new_blocks(self, blocks):
        assert len(self.current_tips) >= 1
        assert (len(blocks) == 1)
        block = blocks[0]

        if len(self.current_tips) == 1:             # No previous fork
            block.link(self.current_tips[0])        # Directly link last block
            self.current_tips.clear()
            self.current_tips.append(block)

        else:                                       # Solve previous fork
            for tip in self.current_tips:
                if tip.miner == block.last_miner:   # Find matched last block in competing tips
                    block.link(tip)
                    break
            self.current_tips.clear()
            self.current_tips.append(block)


    def new_competing_blocks(self, compete_blocks):
        assert len(self.current_tips) >= 1

        self.fork_count += 1
        tip_assign = True

        if len(self.current_tips) == 1:             # No previous fork
            tip = self.current_tips[0]
            self.current_tips.clear()

            for blocks in compete_blocks:
                assert (len(blocks) == 1 or len(blocks) == 2)
                blocks[0].link(tip)                 # Link tips

                if tip_assign:                      # Update current_tips
                    if len(blocks) == 1:
                        self.current_tips.append(blocks[0])
                    elif len(blocks) == 2:          # 2 consecutive blocks
                        tip_assign = False
                        blocks[1].link(blocks[0])
                        self.current_tips.clear()
                        self.current_tips.append(blocks[1])

                self.forks[blocks[0].miner] += 1

        else:                                       # Solve previous fork
            for blocks in compete_blocks:
                assert (len(blocks) == 1 or len(blocks) == 2)
                for tip in self.current_tips:
                    if tip.miner == blocks[0].last_miner:
                        blocks[0].link(tip)
            self.current_tips.clear()

            for blocks in compete_blocks:           # Update current_tips
                if tip_assign:
                    if len(blocks) == 1:
                        self.current_tips.append(blocks[0])
                    elif len(blocks) == 2:
                        tip_assign = False
                        blocks[1].link(blocks[0])
                        self.current_tips.clear()
                        self.current_tips.append(blocks[1])

                self.forks[blocks[0].miner] += 1

    def get_competitors(self):
        competitors = []
        for block in self.current_tips:
            competitors.append(block.miner)

        return competitors

    def get_revenue(self, checkpoint=0):
        revenue = np.ones(self.num_node, dtype=int)
        block = self.current_tips[0]
        while block.last_block is not None and block.height >= checkpoint:
            revenue[block.miner] += 1
            block = block.last_block
        return revenue

    def get_interval(self):
        block = self.current_tips[0]
        return block.timestamp / block.height

    def current_height(self):
        return self.current_tips[0].height
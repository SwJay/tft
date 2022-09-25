# TFT

This repository contains the technical report and code for paper "Fairness Matters: A Tit-For-Tat Strategy Against Selfish Mining".

## Usage

```
usage: app.py [-h] [--tft TFT] [--sm] [--writegraph] [--writepower]
              [--identical] [--nodes NODES] [--rounds ROUNDS] [--batch BATCH]
              [--alpha ALPHA] [--gamma GAMMA] [--mediantime MEDIANTIME]
              [--cv CV] [--step STEP] [--write] [--case CASE]
              [--session SESSION] [--smsession SMSESSION] [--length LENGTH]
              [--mature MATURE] [--powrange POWRANGE] [--powcycle POWCYCLE]
              [--disrange DISRANGE] [--discycle DISCYCLE] [--oracle]

optional arguments:
  -h, --help            show this help message and exit
  --tft TFT, -tft TFT   apply tit-for-tat strategy
  --sm, -sm             sm mode
  --writegraph, -wg     generate new graph and write in disk
  --writepower, -wp     generate new power distribution and write in disk
  --identical, -i       nodes have identical computing power
  --nodes NODES, -n NODES
                        total number of nodes
  --rounds ROUNDS, -r ROUNDS
                        total rounds
  --batch BATCH, -b BATCH
                        batch test
  --alpha ALPHA, -a ALPHA
                        alpha for sm
  --gamma GAMMA, -g GAMMA
                        gamma for sm
  --mediantime MEDIANTIME, -mt MEDIANTIME
                        median time
  --cv CV, -cv CV       cv for sigma
  --step STEP, -s STEP  step scale
  --write, -w           write result
  --case CASE, -c CASE  pre generated graph/power
  --session SESSION, -ss SESSION
                        session length in peer dynamic (churn)
  --smsession SMSESSION, -smss SMSESSION
                        session length in sm churn
  --length LENGTH, -l LENGTH
                        window length
  --mature MATURE, -m MATURE
                        mature length
  --powrange POWRANGE, -pr POWRANGE
                        power range
  --powcycle POWCYCLE, -pc POWCYCLE
                        power cycle
  --oracle, -o          static oracle
```

Example of usage:

```
Python3 app.py -a 0.3 -g 0.7 -c 1 -sm -tft 2 -pr 0.2 -pc 20 -r 1000
```

- default values:
  - `tft`: info.Tft.NONE.value
  - `sm` : False
  - `writegraph`: False
  - `writepower`: False
  - `identical`: False
  - `nodes`: 16
  - `rounds`: 10000
  - `batch`: info.Test.NONE.value
  - `alpha`: 0
  - `gamma`: 0
  - `mediantime`: 8.7
  - `cv`: 0.02
  - `step`: 3
  - `write`: False
  - `case`: -1
  - `session`: np.infty
  - `smsession`: np.infty
  - `length`: 100
  - `mature`: 50
  - `powrange`: 0
  - `powcycle`: np.infty
  - `oracle`: False

## Reproduce of results

```
python3 app.py [-b NO.CASE]
```

## Graph generation

```
python3 gen_network.py
```


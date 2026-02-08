# Target Design Profile: AI Accelerators

## Domain Focus
- **Primary Target**: AI accelerators with dense datapaths and brutal congestion
- **Sub-category**: Matrix multiplication units, convolution cores, tensor processing engines
- **Process Node**: Initially 7nm, extensible to 5nm and 3nm

## Typical Block Sizes
- **Small**: 1-2mm² (32x32 MAC arrays, simple convolution units)
- **Medium**: 3-5mm² (64x64 MAC arrays, multi-channel convolutions)
- **Large**: 6-10mm² (tensor processing units, multi-core AI blocks)

## Critical Failure Modes
1. **Congestion**: Dense compute units create routing bottlenecks
2. **Timing**: Critical paths through compute elements violate slack
3. **Power**: Hotspots in compute clusters exceed thermal limits
4. **DRC**: Density rules violated in high-utilization regions

## PPA Priorities
1. **Performance**: Maximize throughput and frequency
2. **Area**: Minimize die size within performance constraints  
3. **Power**: Manage thermal profiles and leakage

## What Humans Usually Screw Up
- Underestimating interconnect requirements in dense compute
- Poor floorplanning of memory-to-compute pathways
- Insufficient consideration of simultaneous switching noise
- Inadequate power distribution for peak compute demands

## Success Metrics
- **Congestion Prediction Accuracy**: >85% correlation with actual results
- **Timing Closure Success**: >70% first-pass timing closure
- **Power Hotspot Prediction**: >80% identification of thermal issues
- **DRC Violation Prediction**: >75% prediction of post-route violations
- **Iteration Reduction**: 50% reduction in design iterations

## Risk Indicators
- High fanout nets (>100 loads) in critical paths
- Wide buses (64+ bit) through congested regions
- Clock domains crossing power islands
- Asynchronous interfaces between compute units

## Oracle Enhancement Requirements
- Predict congestion "heat maps" before placement
- Identify timing-critical paths pre-layout
- Flag power density hotspots early
- Anticipate DRC rule violations based on topology
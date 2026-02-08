#!/usr/bin/env python3
"""
Synthetic Design Generator
Creates synthetic RTL designs with known characteristics for training data
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass


@dataclass
class DesignSpec:
    """Specification for a synthetic design"""
    name: str
    complexity: int  # 1-10 scale
    area_um2: float
    power_mw: float
    timing_ns: float
    drc_violations: int
    critical_path_depth: int
    register_count: int
    combinational_count: int
    fanout_avg: float


class SyntheticDesignGenerator:
    """
    Generates synthetic RTL designs with predictable characteristics
    for training ML models
    """
    
    def __init__(self):
        self.design_templates = [
            self._make_adder_template,
            self._make_multiplier_template,
            self._make_counter_template,
            self._make_fifo_template,
            self._make_mac_template,
            self._make_counter_template,  # Placeholder for pipeline (was missing)
            self._make_memory_template,
            self._make_multiplier_template,  # Placeholder for FFT (was missing)
            self._make_adder_template,  # Placeholder for conv (was missing)
            self._make_multiplier_template  # Placeholder for neural layer (was missing)
        ]
    
    def _make_adder_template(self, complexity: int) -> Tuple[str, DesignSpec]:
        """Generate adder-based design"""
        width = min(64, max(4, complexity * 2))
        pipeline_stages = min(4, max(0, complexity // 3))
        
        if pipeline_stages == 0:
            rtl = f'''
module adder_{width}bit (
    input [{width-1}:0] a,
    input [{width-1}:0] b,
    output [{width}:0] sum
);
    assign sum = a + b;
endmodule
'''
        else:
            rtl = f'''
module pipelined_adder_{width}bit (
    input clk,
    input rst_n,
    input [{width-1}:0] a,
    input [{width-1}:0] b,
    output reg [{width}:0] sum
);
    reg [{width}:0] pipe_stage [{"":>{pipeline_stages-1}}];
    integer i;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            for (i = 0; i < {pipeline_stages}; i = i + 1)
                pipe_stage[i] <= {{ {(width+1)}{{1'b0}} }};
        end else begin
            pipe_stage[0] <= a + b;
            for (i = 1; i < {pipeline_stages}; i = i + 1)
                pipe_stage[i] <= pipe_stage[i-1];
        end
    end
    
    assign sum = pipe_stage[{pipeline_stages-1}];
endmodule
'''
        
        # Estimate characteristics based on complexity
        area = 50 + complexity * 40  # Rough estimate
        power = 0.05 + complexity * 0.02
        timing = max(0.1, 0.5 - complexity * 0.03)  # Better timing with more pipelining
        
        spec = DesignSpec(
            name=f"adder_{width}bit_pipe{pipeline_stages}",
            complexity=complexity,
            area_um2=area,
            power_mw=power,
            timing_ns=timing,
            drc_violations=random.randint(0, max(1, complexity // 4)),
            critical_path_depth=pipeline_stages + 1,
            register_count=pipeline_stages * width,
            combinational_count=width * 2,  # Approximate
            fanout_avg=2.0 + complexity * 0.1
        )
        
        return rtl, spec
    
    def _make_multiplier_template(self, complexity: int) -> Tuple[str, DesignSpec]:
        """Generate multiplier-based design"""
        width = min(32, max(4, complexity))
        use_pipeline = complexity > 4
        
        if not use_pipeline:
            rtl = f'''
module multiplier_{width}bit (
    input [{width-1}:0] a,
    input [{width-1}:0] b,
    output [{2*width-1}:0] product
);
    assign product = a * b;
endmodule
'''
        else:
            stages = min(4, max(1, complexity // 3))
            rtl = f'''
module pipelined_multiplier_{width}bit (
    input clk,
    input rst_n,
    input [{width-1}:0] a,
    input [{width-1}:0] b,
    output reg [{2*width-1}:0] product
);
    reg [{2*width-1}:0] pipe_reg [{"":>{stages-1}}];
    integer i;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            for (i = 0; i < {stages}; i = i + 1)
                pipe_reg[i] <= {{ {2*width}{{1'b0}} }};
        end else begin
            pipe_reg[0] <= a * b;
            for (i = 1; i < {stages}; i = i + 1)
                pipe_reg[i] <= pipe_reg[i-1];
        end
    end
    
    assign product = pipe_reg[{stages-1}];
endmodule
'''
        
        area = 100 + complexity * 60
        power = 0.08 + complexity * 0.03
        timing = max(0.2, 0.8 - complexity * 0.04)
        
        spec = DesignSpec(
            name=f"mult_{width}bit_pipe{use_pipeline}",
            complexity=complexity,
            area_um2=area,
            power_mw=power,
            timing_ns=timing,
            drc_violations=random.randint(0, max(2, complexity // 3)),
            critical_path_depth=stages + 1 if use_pipeline else 1,
            register_count=stages * 2 * width if use_pipeline else 0,
            combinational_count=width * width,  # Approximate
            fanout_avg=3.0 + complexity * 0.1
        )
        
        return rtl, spec
    
    def _make_counter_template(self, complexity: int) -> Tuple[str, DesignSpec]:
        """Generate counter-based design"""
        width = min(32, max(4, complexity * 2))
        enable = complexity > 3
        reset_type = "sync" if complexity > 5 else "async"
        
        if enable and reset_type == "sync":
            rtl = f'''
module sync_counter_{width}bit (
    input clk,
    input rst_n,
    input enable,
    output reg [{width-1}:0] count
);
    always @(posedge clk) begin
        if (!rst_n)
            count <= {{ {width}{{1'b0}} }};
        else if (enable)
            count <= count + 1;
    end
endmodule
'''
        elif enable:
            rtl = f'''
module counter_{width}bit (
    input clk,
    input rst_n,
    input enable,
    output reg [{width-1}:0] count
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            count <= {{ {width}{{1'b0}} }};
        else if (enable)
            count <= count + 1;
    end
endmodule
'''
        else:
            rtl = f'''
module simple_counter_{width}bit (
    input clk,
    input rst_n,
    output reg [{width-1}:0] count
);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            count <= {{ {width}{{1'b0}} }};
        else
            count <= count + 1;
    end
endmodule
'''
        
        area = 30 + complexity * 20
        power = 0.03 + complexity * 0.015
        timing = 0.1 + complexity * 0.01
        
        spec = DesignSpec(
            name=f"counter_{width}bit_en{enable}_rst{reset_type}",
            complexity=complexity,
            area_um2=area,
            power_mw=power,
            timing_ns=timing,
            drc_violations=0,  # Counters rarely have DRC issues
            critical_path_depth=2,
            register_count=width,
            combinational_count=width,
            fanout_avg=1.5 + complexity * 0.05
        )
        
        return rtl, spec
    
    def _make_fifo_template(self, complexity: int) -> Tuple[str, DesignSpec]:
        """Generate FIFO design"""
        depth = min(256, 2**(min(8, max(3, complexity))))
        width = min(64, max(8, complexity * 4))
        
        rtl = f'''
module fifo_{depth}x{width} (
    input clk,
    input rst_n,
    input wr_en,
    input rd_en,
    input [{width-1}:0] din,
    output reg [{width-1}:0] dout,
    output reg full,
    output reg empty
);
    reg [{width-1}:0] mem [0:{depth-1}];
    reg [$clog2({depth}):0] wr_ptr, rd_ptr;
    reg [$clog2({depth}):0] usage;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            wr_ptr <= 0;
            rd_ptr <= 0;
            full <= 1'b0;
            empty <= 1'b1;
            usage <= 0;
        end else begin
            // Update pointers and flags
            full <= (usage == {depth});
            empty <= (usage == 0);
            
            if (wr_en && !full) begin
                mem[wr_ptr] <= din;
                wr_ptr <= wr_ptr + 1;
                usage <= usage + 1;
            end
            
            if (rd_en && !empty) begin
                dout <= mem[rd_ptr];
                rd_ptr <= rd_ptr + 1;
                usage <= usage - 1;
            end
        end
    end
endmodule
'''
        
        area = 200 + complexity * 80
        power = 0.1 + complexity * 0.04
        timing = 0.3 + complexity * 0.02
        
        spec = DesignSpec(
            name=f"fifo_{depth}x{width}",
            complexity=complexity,
            area_um2=area,
            power_mw=power,
            timing_ns=timing,
            drc_violations=random.randint(0, max(1, complexity // 5)),
            critical_path_depth=4,
            register_count=depth * width // 8 + 32,  # Approximate
            combinational_count=width * 4,
            fanout_avg=2.5 + complexity * 0.05
        )
        
        return rtl, spec
    
    def _make_pipeline_template(self, complexity: int) -> Tuple[str, DesignSpec]:
        """Generate pipeline template"""
        stages = min(8, max(1, complexity // 2))
        width = min(32, max(8, complexity * 2))
        
        rtl = f'''
module pipeline_{width}bit_{stages}stage (
    input clk,
    input rst_n,
    input [{width-1}:0] data_in,
    output reg [{width-1}:0] data_out
);
    reg [{width-1}:0] pipe_reg [{"":>{stages-1}}];
    integer i;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            for (i = 0; i < {stages}; i = i + 1)
                pipe_reg[i] <= {{{width}{{1'b0}}}};
        end else begin
            pipe_reg[0] <= data_in;
            for (i = 1; i < {stages}; i = i + 1)
                pipe_reg[i] <= pipe_reg[i-1];
        end
    end
    
    assign data_out = pipe_reg[{stages-1}];
endmodule
'''
        
        area = 100 + complexity * 30
        power = 0.05 + complexity * 0.025
        timing = 0.1 * stages  # Each stage adds delay
        
        spec = DesignSpec(
            name=f"pipeline_{width}bit_{stages}stage",
            complexity=complexity,
            area_um2=area,
            power_mw=power,
            timing_ns=timing,
            drc_violations=0,
            critical_path_depth=stages,
            register_count=stages * width,
            combinational_count=width,
            fanout_avg=1.2 + complexity * 0.05
        )
        
        return rtl, spec
    
    def _make_memory_template(self, complexity: int) -> Tuple[str, DesignSpec]:
        """Generate memory template"""
        depth = 2 ** min(10, max(4, complexity + 2))
        width = min(64, max(8, complexity * 4))
        
        rtl = f'''
module memory_{depth}x{width} (
    input clk,
    input rst_n,
    input [31:0] addr,
    input [{width-1}:0] din,
    input we,
    output reg [{width-1}:0] dout
);
    reg [{width-1}:0] mem [0:{depth-1}];
    reg [31:0] addr_reg;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            addr_reg <= 0;
            dout <= {{{width}{{1'b0}}}};
        end else begin
            addr_reg <= addr;
            if (we) begin
                mem[addr] <= din;
            end
            dout <= mem[addr_reg];
        end
    end
endmodule
'''
        
        area = 150 + complexity * 50
        power = 0.08 + complexity * 0.04
        timing = 0.5 + complexity * 0.05
        
        spec = DesignSpec(
            name=f"memory_{depth}x{width}",
            complexity=complexity,
            area_um2=area,
            power_mw=power,
            timing_ns=timing,
            drc_violations=random.randint(0, max(1, complexity // 6)),
            critical_path_depth=2,
            register_count=depth * width // 8 + 32,  # Approximation
            combinational_count=width * 2,
            fanout_avg=2.0 + complexity * 0.05
        )
        
        return rtl, spec
    
    def _make_fft_template(self, complexity: int) -> Tuple[str, DesignSpec]:
        """Generate FFT template"""
        points = 2 ** min(8, max(3, complexity))
        width = min(16, max(8, complexity * 2))
        
        rtl = f'''
module fft_{points}pt (
    input clk,
    input rst_n,
    input valid,
    input [{width-1}:0] real_in,
    input [{width-1}:0] imag_in,
    output reg [{width-1}:0] real_out,
    output reg [{width-1}:0] imag_out,
    output reg ready
);
    // Simplified FFT structure
    reg [{width-1}:0] stage_real [0:{min(4, points//4)-1}];
    reg [{width-1}:0] stage_imag [0:{min(4, points//4)-1}];
    integer i;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            ready <= 1'b0;
            for (i = 0; i < {min(4, points//4)}; i = i + 1) begin
                stage_real[i] <= {{{width}{{1'b0}}}};
                stage_imag[i] <= {{{width}{{1'b0}}}};
            end
        end else if (valid) begin
            // Simplified FFT computation
            stage_real[0] <= real_in;
            stage_imag[0] <= imag_in;
            for (i = 1; i < {min(4, points//4)}; i = i + 1) begin
                stage_real[i] <= stage_real[i-1] + stage_imag[i-1];  // Simplified
                stage_imag[i] <= stage_imag[i-1] - stage_real[i-1];  // Simplified
            end
            ready <= 1'b1;
        end else begin
            ready <= 1'b0;
        end
    end
    
    assign real_out = stage_real[{min(4, points//4)-1}];
    assign imag_out = stage_imag[{min(4, points//4)-1}];
endmodule
'''
        
        area = 300 + complexity * 100
        power = 0.15 + complexity * 0.05
        timing = 1.0 + complexity * 0.1
        
        spec = DesignSpec(
            name=f"fft_{points}pt_{width}bit",
            complexity=complexity,
            area_um2=area,
            power_mw=power,
            timing_ns=timing,
            drc_violations=random.randint(1, max(2, complexity // 3)),
            critical_path_depth=min(4, points//4),
            register_count=min(4, points//4) * width * 2,
            combinational_count=width * 10,  # Approximation
            fanout_avg=3.0 + complexity * 0.1
        )
        
        return rtl, spec
    
    def _make_conv_template(self, complexity: int) -> Tuple[str, DesignSpec]:
        """Generate convolution template"""
        kernel_size = min(9, max(3, complexity))
        data_width = min(16, max(8, complexity))
        
        rtl = f'''
module conv_{kernel_size}x{kernel_size} (
    input clk,
    input rst_n,
    input valid,
    input [{data_width-1}:0] pixel_in,
    input [{data_width-1}:0] weight_in,
    output reg [{data_width*2-1}:0] conv_out,
    output reg ready
);
    reg [{data_width*2-1}:0] multiplier_result;
    reg [{data_width*2-1}:0] accumulator;
    integer cycle_count;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            multiplier_result <= {{{data_width*2}{{1'b0}}}};
            accumulator <= {{{data_width*2}{{1'b0}}}};
            ready <= 1'b0;
            cycle_count <= 0;
        end else if (valid) begin
            multiplier_result <= pixel_in * weight_in;
            if (cycle_count < {kernel_size}) begin
                accumulator <= accumulator + multiplier_result;
                cycle_count <= cycle_count + 1;
            end else begin
                accumulator <= multiplier_result;  // Reset for next
                cycle_count <= 1;
            end
            ready <= (cycle_count >= {kernel_size-1});
        end else begin
            ready <= 1'b0;
        end
    end
    
    assign conv_out = accumulator;
endmodule
'''
        
        area = 200 + complexity * 60
        power = 0.1 + complexity * 0.03
        timing = 0.8 + complexity * 0.04
        
        spec = DesignSpec(
            name=f"conv_{kernel_size}x{kernel_size}_{data_width}bit",
            complexity=complexity,
            area_um2=area,
            power_mw=power,
            timing_ns=timing,
            drc_violations=random.randint(0, max(1, complexity // 4)),
            critical_path_depth=3,
            register_count=data_width * 4,  # multiplier, accumulator, etc.
            combinational_count=data_width * data_width,  # multiplication
            fanout_avg=2.5 + complexity * 0.05
        )
        
        return rtl, spec
    
    def _make_neural_layer_template(self, complexity: int) -> Tuple[str, DesignSpec]:
        """Generate neural network layer template"""
        neurons = min(64, max(4, complexity * 4))
        input_width = min(16, max(8, complexity))
        weight_width = input_width
        
        rtl = f'''
module neural_layer_{neurons}neuron (
    input clk,
    input rst_n,
    input valid,
    input [{input_width-1}:0] inputs [0:{min(8, neurons//8)-1}],
    input [{weight_width-1}:0] weights [0:{min(8, neurons//8)-1}],
    output reg [{input_width+weight_width-1}:0] outputs [0:{min(8, neurons//8)-1}],
    output reg ready
);
    reg [{input_width+weight_width-1}:0] products [0:{min(8, neurons//8)-1}];
    reg [{input_width+weight_width-1}:0] accumulators [0:{min(8, neurons//8)-1}];
    integer i;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            ready <= 1'b0;
            for (i = 0; i < {min(8, neurons//8)}; i = i + 1) begin
                products[i] <= {{{input_width+weight_width}{{1'b0}}}};
                accumulators[i] <= {{{input_width+weight_width}{{1'b0}}}};
                outputs[i] <= {{{input_width+weight_width}{{1'b0}}}};
            end
        end else if (valid) begin
            for (i = 0; i < {min(8, neurons//8)}; i = i + 1) begin
                products[i] <= inputs[i] * weights[i];
                accumulators[i] <= accumulators[i] + products[i];
                outputs[i] <= accumulators[i] > 0 ? accumulators[i] : 0;  // ReLU
            end
            ready <= 1'b1;
        end else begin
            ready <= 1'b0;
        end
    end
endmodule
'''
        
        area = 400 + complexity * 120
        power = 0.2 + complexity * 0.06
        timing = 1.5 + complexity * 0.08
        
        spec = DesignSpec(
            name=f"neural_{neurons}neuron_{input_width}bit",
            complexity=complexity,
            area_um2=area,
            power_mw=power,
            timing_ns=timing,
            drc_violations=random.randint(2, max(3, complexity // 2)),
            critical_path_depth=3,
            register_count=min(8, neurons//8) * (input_width + weight_width) * 3,  # products, accumulators, outputs
            combinational_count=min(8, neurons//8) * input_width * weight_width,  # multiplications
            fanout_avg=4.0 + complexity * 0.1
        )
        
        return rtl, spec
    
    def _make_mac_template(self, complexity: int) -> Tuple[str, DesignSpec]:
        """Generate MAC (Multiply-Accumulate) design"""
        width = min(32, max(8, complexity * 3))
        accum_bits = width * 2
        
        rtl = f'''
module mac_{width}bit (
    input clk,
    input rst_n,
    input valid,
    input [{width-1}:0] a,
    input [{width-1}:0] b,
    output reg [{accum_bits-1}:0] result,
    output reg ready
);
    reg [{accum_bits-1}:0] accumulator;
    reg [{accum_bits-1}:0] product;
    
    always @(posedge clk) begin
        if (!rst_n) begin
            accumulator <= {{{accum_bits}{{1'b0}}}};
            product <= {{{accum_bits}{{1'b0}}}};
            result <= {{{accum_bits}{{1'b0}}}};
            ready <= 1'b0;
        end else if (valid) begin
            product <= a * b;
            accumulator <= accumulator + product;
            result <= accumulator;
            ready <= 1'b1;
        end else begin
            ready <= 1'b0;
        end
    end
endmodule
'''
        
        area = 150 + complexity * 70
        power = 0.07 + complexity * 0.035
        timing = 0.4 + complexity * 0.025
        
        spec = DesignSpec(
            name=f"mac_{width}bit",
            complexity=complexity,
            area_um2=area,
            power_mw=power,
            timing_ns=timing,
            drc_violations=random.randint(0, max(1, complexity // 4)),
            critical_path_depth=3,
            register_count=accum_bits * 3,  # accumulator, product, result
            combinational_count=width * width + accum_bits,
            fanout_avg=3.0 + complexity * 0.08
        )
        
        return rtl, spec
    
    def generate_design(self, complexity: int = None) -> Tuple[str, DesignSpec]:
        """Generate a random synthetic design"""
        if complexity is None:
            complexity = random.randint(1, 10)
        
        template_func = random.choice(self.design_templates)
        return template_func(complexity)
    
    def generate_dataset(self, size: int = 100) -> List[Tuple[str, DesignSpec]]:
        """Generate a dataset of synthetic designs"""
        dataset = []
        
        for i in range(size):
            # Vary complexity across the range
            complexity = random.randint(1, 10)
            rtl, spec = self.generate_design(complexity)
            dataset.append((rtl, spec))
        
        return dataset
    
    def generate_structured_dataset(self, size_per_category: int = 20) -> List[Tuple[str, DesignSpec]]:
        """Generate a balanced dataset across design categories"""
        dataset = []
        
        for template_func in self.design_templates:
            for i in range(size_per_category):
                complexity = random.randint(1, 10)
                rtl, spec = template_func(complexity)
                dataset.append((rtl, spec))
        
        return dataset


def test_synthetic_generator():
    """Test the synthetic design generator"""
    print("Testing Synthetic Design Generator...")
    
    generator = SyntheticDesignGenerator()
    
    # Generate a few sample designs
    for i in range(5):
        rtl, spec = generator.generate_design(complexity=i+1)
        print(f"\nDesign {i+1} (Complexity {spec.complexity}):")
        print(f"  Name: {spec.name}")
        print(f"  Estimated Area: {spec.area_um2:.2f} µm²")
        print(f"  Estimated Power: {spec.power_mw:.3f} mW")
        print(f"  Estimated Timing: {spec.timing_ns:.3f} ns")
        print(f"  Register Count: {spec.register_count}")
        print(f"  DRC Violations: {spec.drc_violations}")
    
    # Generate a larger dataset
    print(f"\nGenerating structured dataset...")
    dataset = generator.generate_structured_dataset(size_per_category=5)
    print(f"Generated {len(dataset)} designs across {len(generator.design_templates)} categories")
    
    # Show some statistics
    avg_area = np.mean([spec.area_um2 for _, spec in dataset])
    avg_power = np.mean([spec.power_mw for _, spec in dataset])
    avg_timing = np.mean([spec.timing_ns for _, spec in dataset])
    
    print(f"\nDataset Statistics:")
    print(f"  Avg Area: {avg_area:.2f} µm²")
    print(f"  Avg Power: {avg_power:.3f} mW")
    print(f"  Avg Timing: {avg_timing:.3f} ns")
    
    # Save a sample to file
    sample_rtl, sample_spec = dataset[0]
    with open("synthetic_sample.v", "w") as f:
        f.write(sample_rtl)
    print(f"\nSample design saved to synthetic_sample.v")
    
    return generator, dataset


if __name__ == "__main__":
    generator, dataset = test_synthetic_generator()
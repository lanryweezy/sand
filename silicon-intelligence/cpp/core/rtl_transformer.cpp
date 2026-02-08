#include "rtl_transformer.hpp"
#include <fstream>
#include <regex>
#include <iostream>

namespace silicon_intelligence {

// Helper to check if string contains substring
bool contains(const std::string& str, const std::string& substr) {
    return str.find(substr) != std::string::npos;
}

// -----------------------------------------------------------------------------
// Parser Implementation
// -----------------------------------------------------------------------------

std::shared_ptr<Module> RTLTransformer::parse_verilog(const std::string& source_code) {
    // Basic Regex-based parser for the demonstration subset of Verilog
    // NOTE: In a production environment, this would use Flex/Bison or ANTLR.
    
    // 1. Find Module Decl
    std::regex module_re(R"(module\s+(\w+)\s*\(([^)]*)\);)");
    std::smatch match;
    
    std::string code = source_code;
    std::string module_name = "unknown_module";
    std::shared_ptr<Module> module;

    if (std::regex_search(code, match, module_re)) {
        module_name = match[1].str();
        module = std::make_shared<Module>(module_name);
        
        std::string port_list_str = match[2].str();
        
        // Parse ports from port list (simplified)
        std::regex port_re(R"((input|output|inout)\s+(?:\[(\d+):(\d+)\]\s+)?(\w+))");
        auto port_begin = std::sregex_iterator(port_list_str.begin(), port_list_str.end(), port_re);
        auto port_end = std::sregex_iterator();
        
        for (std::sregex_iterator i = port_begin; i != port_end; ++i) {
            std::smatch port_match = *i;
            std::string dir_str = port_match[1].str();
            std::string width_msb = port_match[2].str(); // Optional
            std::string width_lsb = port_match[3].str(); // Optional
            std::string name = port_match[4].str();
            
            PortDirection dir = PortDirection::INPUT;
            if (dir_str == "output") dir = PortDirection::OUTPUT;
            else if (dir_str == "inout") dir = PortDirection::INOUT;
            
            int width = 1;
            if (!width_msb.empty()) {
                width = std::stoi(width_msb) - std::stoi(width_lsb) + 1;
            }
            
            module->add_port(std::make_shared<Port>(name, dir, width));
        }
    } else {
        // Fallback for demo if no module found
        module = std::make_shared<Module>("dummy_module");
    }

    // 2. Find Wires/Regs
    std::regex net_re(R"((wire|reg)\s+(?:\[(\d+):(\d+)\]\s+)?(\w+)\s*;)");
    auto net_begin = std::sregex_iterator(code.begin(), code.end(), net_re);
    auto net_end = std::sregex_iterator();
    
    for (std::sregex_iterator i = net_begin; i != net_end; ++i) {
        std::smatch net_match = *i;
        std::string type_str = net_match[1].str();
        std::string width_msb = net_match[2].str();
        std::string width_lsb = net_match[3].str();
        std::string name = net_match[4].str();
        
        NetType type = (type_str == "reg") ? NetType::REG : NetType::WIRE;
        int width = 1;
        if (!width_msb.empty()) {
            width = std::stoi(width_msb) - std::stoi(width_lsb) + 1;
        }
        
        module->add_item(std::make_shared<Net>(name, type, width));
    }

    // 3. Find Assigns
    std::regex assign_re(R"(assign\s+(\w+)\s*=\s*([^;]+);)");
    auto assign_begin = std::sregex_iterator(code.begin(), code.end(), assign_re);
    auto assign_end = std::sregex_iterator();
    
    for (std::sregex_iterator i = assign_begin; i != assign_end; ++i) {
        std::smatch assign_match = *i;
        module->add_item(std::make_shared<Assign>(assign_match[1].str(), assign_match[2].str()));
    }

    // Capture the rest (Always blocks, Instances) implies more complex parsing.
    // For this implementation, we will assume simple always blocks for pipelining are usually
    // generated or simple to parse. We'll skip complex parsing for Phase 3 Proof of Concept.
    
    return module;
}

std::shared_ptr<Module> RTLTransformer::parse_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Could not open file: " << filepath << std::endl;
        return nullptr;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return parse_verilog(buffer.str());
}

// -----------------------------------------------------------------------------
// Transformations
// -----------------------------------------------------------------------------

void RTLTransformer::add_pipeline_stage(std::shared_ptr<Module> module, const std::string& signal_name) {
    // 1. Find the signal to pipe (Port or Net)
    int width = 1;
    bool found = false;
    
    for (const auto& port : module->ports) {
        if (port->name == signal_name) {
            width = port->width;
            found = true;
            break;
        }
    }
    
    if (!found) {
        for (const auto& item : module->items) {
            if (auto net = std::dynamic_pointer_cast<Net>(item)) {
                if (net->name == signal_name) {
                    width = net->width;
                    found = true;
                    break;
                }
            }
        }
    }

    std::string pipe_reg_name = signal_name + "_pipe_reg";
    
    // 2. Add pipe register
    module->add_item(std::make_shared<Net>(pipe_reg_name, NetType::REG, width));
    
    // 3. Add behavior: always @(posedge clk) pipe_reg <= signal;
    auto always = std::make_shared<AlwaysBlock>("posedge clk");
    always->add_statement(pipe_reg_name + " <= " + signal_name + ";");
    module->add_item(always);
    
    // 4. Update usages (Simple Replace in Assigns)
    // We only replace usages where the signal is on the RHS (source)
    for (auto& item : module->items) {
        if (auto assign = std::dynamic_pointer_cast<Assign>(item)) {
            // Very naive replacement: replace substring "signal" with "pipe" in RHS
            // Real implementation requires expression parsing.
             // Avoid self-assignment in case we created one? (Unlikely here)
            size_t pos = 0;
            while ((pos = assign->rhs.find(signal_name, pos)) != std::string::npos) {
                // Check if match is exact word
                bool boundary_start = (pos == 0 || !isalnum(assign->rhs[pos-1]) && assign->rhs[pos-1] != '_');
                bool boundary_end = (pos + signal_name.length() == assign->rhs.length() || !isalnum(assign->rhs[pos + signal_name.length()]) && assign->rhs[pos + signal_name.length()] != '_');
                
                if (boundary_start && boundary_end) {
                    assign->rhs.replace(pos, signal_name.length(), pipe_reg_name);
                    pos += pipe_reg_name.length();
                } else {
                    pos += signal_name.length();
                }
            }
        }
    }
}

void RTLTransformer::insert_clock_gate(std::shared_ptr<Module> module, const std::string& reg_signal, const std::string& enable) {
    std::string gated_clk_name = reg_signal + "_gated_clk";
    
    // 1. Add gated clock wire
    module->add_item(std::make_shared<Net>(gated_clk_name, NetType::WIRE));
    
    // 2. Add ICG instance
    auto icg = std::make_shared<Instance>("sky130_fd_sc_hd__lpflow_is_1", "icg_" + reg_signal);
    icg->connect("CLK", "clk");
    icg->connect("ENA", enable);
    icg->connect("GCLK", gated_clk_name);
    module->add_item(icg);
    
    // 3. Update Always blocks using main clock to use gated clock IF they drive the reg_signal
    // This is hard with the current simple string storage of Always blocks. 
    // We'll append a comment for now or assume a specific structure?
    // Let's iterate and modify the sensitivity list if possible
    
    for (auto& item : module->items) {
        if (auto always = std::dynamic_pointer_cast<AlwaysBlock>(item)) {
            // Check if this block controls our register (naive text check)
            bool controls_reg = false;
            for(const auto& stmt : always->statements) {
                if(contains(stmt, reg_signal + " <=") || contains(stmt, reg_signal + " =")) {
                    controls_reg = true;
                    break;
                }
            }
            
            if (controls_reg) {
                // Replace "clk" with gated clock in sensitivity list
                size_t pos = always->sensitivity_list.find("clk");
                if (pos != std::string::npos) {
                     always->sensitivity_list.replace(pos, 3, gated_clk_name);
                }
            }
        }
    }
}

std::string RTLTransformer::generate_verilog(std::shared_ptr<Module> module) {
    return module->to_verilog();
}

void RTLTransformer::apply_logic_merging(std::shared_ptr<Module> module) {
    std::map<std::string, std::string> rhs_to_lhs;
    std::vector<std::shared_ptr<ASTNode>> new_items;
    std::map<std::string, std::string> replacements;

    for (auto& item : module->items) {
        if (auto assign = std::dynamic_pointer_cast<Assign>(item)) {
            if (rhs_to_lhs.count(assign->rhs)) {
                // Signal already exists, replace usages of this LHS with existing LHS
                replacements[assign->lhs] = rhs_to_lhs[assign->rhs];
            } else {
                rhs_to_lhs[assign->rhs] = assign->lhs;
                new_items.push_back(item);
            }
        } else {
            new_items.push_back(item);
        }
    }

    // Apply replacements across all items
    module->items = new_items;
    for (auto& item : module->items) {
        if (auto assign = std::dynamic_pointer_cast<Assign>(item)) {
            for (auto const& [old_s, new_s] : replacements) {
                size_t pos = 0;
                while ((pos = assign->rhs.find(old_s, pos)) != std::string::npos) {
                    assign->rhs.replace(pos, old_s.length(), new_s);
                    pos += new_s.length();
                }
            }
        }
    }
}

void RTLTransformer::apply_fanout_buffering(std::shared_ptr<Module> module, const std::string& signal_name, int degree) {
    if (degree <= 1) return;

    // Add buffer nets and instances
    for (int i = 0; i < degree; ++i) {
        std::string buf_sig = signal_name + "_buf" + std::to_string(i);
        module->add_item(std::make_shared<Net>(buf_sig, NetType::WIRE));
        
        auto buf = std::make_shared<Instance>("sky130_fd_sc_hd__buf_1", "fanout_buf_" + signal_name + "_" + std::to_string(i));
        buf->connect("A", signal_name);
        buf->connect("X", buf_sig);
        module->add_item(buf);
    }
}

void RTLTransformer::apply_input_isolation(std::shared_ptr<Module> module, const std::vector<std::string>& targets, const std::string& enable) {
    for (const auto& target : targets) {
        std::string isolated_sig = target + "_isolated";
        module->add_item(std::make_shared<Net>(isolated_sig, NetType::WIRE));
        
        auto and_gate = std::make_shared<Instance>("sky130_fd_sc_hd__and2_1", "iso_and_" + target);
        and_gate->connect("A", target);
        and_gate->connect("B", enable);
        and_gate->connect("X", isolated_sig);
        module->add_item(and_gate);
        
        // Update usages of target to isolated_sig
        for (auto& item : module->items) {
            if (auto assign = std::dynamic_pointer_cast<Assign>(item)) {
                size_t pos = 0;
                while ((pos = assign->rhs.find(target, pos)) != std::string::npos) {
                    assign->rhs.replace(pos, target.length(), isolated_sig);
                    pos += isolated_sig.length();
                }
            }
        }
    }
}

} // namespace silicon_intelligence

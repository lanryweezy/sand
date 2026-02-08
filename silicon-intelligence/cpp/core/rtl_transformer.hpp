#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <iostream>
#include <sstream>
#include <algorithm>

namespace silicon_intelligence {

// Enum for Port direction
enum class PortDirection {
    INPUT,
    OUTPUT,
    INOUT
};

// Enum for Net type
enum class NetType {
    WIRE,
    REG
};

// Base AST Node
struct ASTNode {
    virtual ~ASTNode() = default;
    virtual std::string to_verilog() const = 0;
};

// Port definition
struct Port : public ASTNode {
    std::string name;
    PortDirection direction;
    int width; // 1 for scalar

    Port(std::string n, PortDirection d, int w = 1) 
        : name(n), direction(d), width(w) {}

    std::string to_verilog() const override {
        std::stringstream ss;
        switch(direction) {
            case PortDirection::INPUT: ss << "input "; break;
            case PortDirection::OUTPUT: ss << "output "; break;
            case PortDirection::INOUT: ss << "inout "; break;
        }
        if (width > 1) {
            ss << "[" << (width - 1) << ":0] ";
        }
        ss << name;
        return ss.str();
    }
};

// Net definition (Wire/Reg)
struct Net : public ASTNode {
    std::string name;
    NetType type;
    int width;

    Net(std::string n, NetType t, int w = 1) 
        : name(n), type(t), width(w) {}

    std::string to_verilog() const override {
        std::stringstream ss;
        switch(type) {
            case NetType::WIRE: ss << "wire "; break;
            case NetType::REG: ss << "reg "; break;
        }
        if (width > 1) {
            ss << "[" << (width - 1) << ":0] ";
        }
        ss << name << ";";
        return ss.str();
    }
};

// Assignment (Continuous)
struct Assign : public ASTNode {
    std::string lhs;
    std::string rhs;

    Assign(std::string l, std::string r) : lhs(l), rhs(r) {}

    std::string to_verilog() const override {
        return "assign " + lhs + " = " + rhs + ";";
    }
};

// Always Block (simplified for behaviors)
struct AlwaysBlock : public ASTNode {
    std::string sensitivity_list; // e.g., "posedge clk or negedge rst_n"
    std::vector<std::string> statements; // Raw strings for body currently

    AlwaysBlock(std::string sens) : sensitivity_list(sens) {}

    void add_statement(std::string stmt) {
        statements.push_back(stmt);
    }

    std::string to_verilog() const override {
        std::stringstream ss;
        ss << "always @(" << sensitivity_list << ") begin\n";
        for (const auto& stmt : statements) {
            ss << "    " << stmt << "\n";
        }
        ss << "end";
        return ss.str();
    }
};

// Module Instance
struct Instance : public ASTNode {
    std::string module_name;
    std::string instance_name;
    std::map<std::string, std::string> port_connections; // .port(signal)

    Instance(std::string mod, std::string inst) : module_name(mod), instance_name(inst) {}

    void connect(std::string port, std::string signal) {
        port_connections[port] = signal;
    }

    std::string to_verilog() const override {
        std::stringstream ss;
        ss << module_name << " " << instance_name << " (\n";
        size_t i = 0;
        for (const auto& pair : port_connections) {
            ss << "    ." << pair.first << "(" << pair.second << ")";
            if (i < port_connections.size() - 1) ss << ",";
            ss << "\n";
            i++;
        }
        ss << ");";
        return ss.str();
    }
};

// Module definition
struct Module : public ASTNode {
    std::string name;
    std::vector<std::shared_ptr<Port>> ports;
    std::vector<std::shared_ptr<ASTNode>> items; // Nets, Assigns, Always, Instances

    Module(std::string n) : name(n) {}

    void add_port(std::shared_ptr<Port> p) { ports.push_back(p); }
    void add_item(std::shared_ptr<ASTNode> item) { items.push_back(item); }

    std::string to_verilog() const override {
        std::stringstream ss;
        ss << "module " << name << " (\n";
        for (size_t i = 0; i < ports.size(); ++i) {
            ss << "    " << ports[i]->to_verilog();
            if (i < ports.size() - 1) ss << ",";
            ss << "\n";
        }
        ss << ");\n\n";
        
        for (const auto& item : items) {
            ss << "    " << item->to_verilog() << "\n";
        }
        
        ss << "\nendmodule";
        return ss.str();
    }
};

class RTLTransformer {
public:
    RTLTransformer() = default;

    // Parsing (Simple Recursive Descent / Regex based for now)
    std::shared_ptr<Module> parse_verilog(const std::string& source_code);
    std::shared_ptr<Module> parse_file(const std::string& filepath);

    // Transformations
    void add_pipeline_stage(std::shared_ptr<Module> module, const std::string& signal_name);
    void insert_clock_gate(std::shared_ptr<Module> module, const std::string& reg_signal, const std::string& enable);
    
    // Code Gen
    std::string generate_verilog(std::shared_ptr<Module> module);

    // Advanced Optimization Algorithms
    void apply_logic_merging(std::shared_ptr<Module> module);
    void apply_fanout_buffering(std::shared_ptr<Module> module, const std::string& signal_name, int degree);
    void apply_input_isolation(std::shared_ptr<Module> module, const std::vector<std::string>& targets, const std::string& enable);
};

} // namespace silicon_intelligence

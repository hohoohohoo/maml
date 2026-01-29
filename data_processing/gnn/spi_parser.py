#!/usr/bin/env python
# coding: utf-8

"""
SPI Parser - TSMC SPICE Netlist Parser
======================================

Parser for TSMC SPI files that extracts logic cell definitions with
transistor connectivity and resistance-based connections.

TSMC SPI format differences from ASAP7 CDL:
- MOS: XMx M1:DRN M1:GATE M1:SRC M1:BULK nch_mac ... w=0.14u ...
- Type: nch_mac (NMOS), pch_mac (PMOS)
- Connections: Defined via resistance (Rx node1 node2 value)

Usage:
    from spi_parser import SPIParser

    parser = SPIParser('path/to/file.spi')
    cells = parser.logic_cells  # Dict of {cell_name: LogicCell}
"""

import re
from collections import namedtuple

# Data structures for parsed SPI data
Transistor = namedtuple('Transistor', ['name', 'drain', 'gate', 'source', 'bulk', 'type', 'width', 'length'])
Resistor = namedtuple('Resistor', ['name', 'node1', 'node2', 'value'])
LogicCell = namedtuple('LogicCell', ['name', 'ports', 'transistors', 'resistors', 'connections'])


class SPIParser:
    """
    TSMC SPI file parser.

    Parses TSMC SPICE SPI files and extracts logic cell definitions with
    transistor information and resistance-based connections.

    Attributes:
        logic_cells: Dictionary of parsed logic cells {cell_name: LogicCell}
    """

    def __init__(self, spi_file_path, verbose=False):
        """
        Parse SPI file and extract logic cells.

        Args:
            spi_file_path: Path to TSMC SPI file
            verbose: Print detailed parsing info
        """
        self.spi_file_path = spi_file_path
        self.logic_cells = {}
        self.verbose = verbose
        self._parse_all_cells()

    def _parse_all_cells(self):
        """Parse all logic cells from SPI file."""
        with open(self.spi_file_path, 'r') as f:
            content = f.read()

        # Pattern: .subckt cell_name ports ... .ends
        subckt_pattern = r'\.subckt\s+(\w+)\s+([^\n]+)\n(.*?)\.ends'
        matches = re.findall(subckt_pattern, content, re.DOTALL | re.IGNORECASE)

        for cell_name, ports_str, body in matches:
            # Only parse logic cells (skip other subcircuits)
            if not self._is_logic_cell(cell_name):
                continue

            # Parse ports, transistors, and resistors
            ports = ports_str.strip().split()
            transistors = self._parse_transistors(body)
            resistors = self._parse_resistors(body)

            # Build connection map from resistors
            connections = self._build_connections(resistors)

            # Store parsed cell
            self.logic_cells[cell_name] = LogicCell(
                name=cell_name,
                ports=ports,
                transistors=transistors,
                resistors=resistors,
                connections=connections
            )

        print(f"   Parsed {len(self.logic_cells)} logic cells from SPI")

    def _is_logic_cell(self, cell_name):
        """
        Check if subcircuit is a logic cell.

        Args:
            cell_name: Subcircuit name

        Returns:
            True if logic cell, False otherwise
        """
        # TSMC cell naming: ends with BWP30P140 or similar
        # Basic logic gate keywords
        basic_keywords = [
            'INV', 'ND', 'AN', 'OR', 'XOR', 'NR', 'XNOR',
            'MUX', 'BUF', 'HA', 'FA', 'MAJ',
            'AOI', 'OAI', 'AO', 'OA',
            'CKND', 'CKAN', 'CKNR',  # Clock cells
            'DEL',  # Delay cells
            'DFCNQD', 'SDFSNQD',  # Flip-flop cells
        ]

        # Complex AO/OA patterns with regex
        ao_oa_patterns = [
            r'AO\d+',        # AO21, AO211, AO22, etc.
            r'OA\d+',        # OA21, OA211, OA22, etc.
            r'AOI\d+',       # AOI21, AOI211, AOI22, etc.
            r'OAI\d+',       # OAI21, OAI211, OAI22, etc.
            r'A\d+O\d+',     # A2O1, A3O2, etc.
            r'O\d+A\d+',     # O2A1, O3A2, etc.
        ]

        # Check basic keywords first
        cell_upper = cell_name.upper()
        if any(keyword in cell_upper for keyword in basic_keywords):
            return True

        # Check complex AO/OA patterns with regex
        for pattern in ao_oa_patterns:
            if re.search(pattern, cell_upper):
                return True

        return False

    def _parse_transistors(self, circuit_body):
        """
        Parse transistor definitions from subcircuit body.

        TSMC SPI MOS format:
        XM1 M1:DRN M1:GATE M1:SRC M1:BULK nch_mac ... w=0.14u ... l=0.03u ...

        Args:
            circuit_body: Text body of subcircuit definition

        Returns:
            List of Transistor namedtuples
        """
        transistors = []

        # Pattern for TSMC SPI MOS:
        # XMx DRN GATE SRC BULK type ...params...
        # Type is nch_mac (NMOS) or pch_mac (PMOS)
        mos_pattern = r'(XM\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(nch_mac|pch_mac)'

        for line in circuit_body.split('\n'):
            line = line.strip()

            # Skip empty lines, comments, and directives
            if not line or line.startswith('.') or line.startswith('*'):
                continue

            match = re.match(mos_pattern, line, re.IGNORECASE)
            if match:
                name = match.group(1)
                drain = match.group(2)
                gate = match.group(3)
                source = match.group(4)
                bulk = match.group(5)
                mos_type = match.group(6)

                # Extract width (w=...) from the rest of the line
                width = self._extract_param(line, 'w')
                length = self._extract_param(line, 'l')

                # Normalize type to pmos/nmos
                normalized_type = 'pmos' if 'pch' in mos_type.lower() else 'nmos'

                transistors.append(Transistor(
                    name=name,
                    drain=drain,
                    gate=gate,
                    source=source,
                    bulk=bulk,
                    type=normalized_type,
                    width=width,
                    length=length
                ))

        if self.verbose:
            print(f"      Found {len(transistors)} transistors")

        return transistors

    def _parse_resistors(self, circuit_body):
        """
        Parse resistor definitions from subcircuit body.

        TSMC SPI Resistor format:
        R1 N_15 M2:SRC 0.001
        R19 M1:SRC VSS:1 9.93124

        Resistors define node connections in TSMC SPI format.

        Args:
            circuit_body: Text body of subcircuit definition

        Returns:
            List of Resistor namedtuples
        """
        resistors = []

        # Pattern: Rx node1 node2 value
        # Nodes can be: N_15, M2:SRC, M1:DRN, VSS:1, VDD, A1, Z, etc.
        res_pattern = r'^(R\d+)\s+(\S+)\s+(\S+)\s+([\d.eE+-]+)'

        for line in circuit_body.split('\n'):
            line = line.strip()

            # Skip empty lines, comments, and directives
            if not line or line.startswith('.') or line.startswith('*'):
                continue

            match = re.match(res_pattern, line)
            if match:
                name = match.group(1)
                node1 = match.group(2)
                node2 = match.group(3)
                value = float(match.group(4))

                resistors.append(Resistor(
                    name=name,
                    node1=node1,
                    node2=node2,
                    value=value
                ))

        if self.verbose:
            print(f"      Found {len(resistors)} resistors")

        return resistors

    def _build_connections(self, resistors):
        """
        Build connection map from resistor information.

        Resistors in TSMC SPI represent physical connections between nodes.
        This function groups connected nodes into equivalence classes.

        Example:
            R1 N_14 M2:SRC 0.001  -> N_14 connected to M2:SRC
            R2 N_14 M1:DRN 7.26785 -> N_14 connected to M1:DRN
            => M2:SRC, M1:DRN, N_14 are all connected

        Args:
            resistors: List of Resistor namedtuples

        Returns:
            dict: Connection map {node: set of connected nodes}
        """
        # Build adjacency list
        connections = {}

        for res in resistors:
            node1, node2 = res.node1, res.node2

            if node1 not in connections:
                connections[node1] = set()
            if node2 not in connections:
                connections[node2] = set()

            connections[node1].add(node2)
            connections[node2].add(node1)

        return connections

    def _extract_param(self, line, param_name):
        """
        Extract parameter value from line.

        Args:
            line: Line containing parameters
            param_name: Parameter name (e.g., 'w', 'l')

        Returns:
            Parameter value in nm, or None if not found
        """
        # First try to find value with unit (e.g., w=0.14u, l=0.03u)
        pattern_with_unit = rf'{param_name}=([\d.eE+-]+)([un])'
        match = re.search(pattern_with_unit, line, re.IGNORECASE)

        if match:
            value = float(match.group(1))
            unit = match.group(2).lower()

            # Convert to nm
            if unit == 'u':
                value *= 1000  # um to nm
            elif unit == 'n':
                pass  # already in nm

            return value

        # Fallback: value without unit (skip w=0 style dummy values)
        pattern_no_unit = rf'{param_name}=([\d.eE+-]+)(?![un\d])'
        match = re.search(pattern_no_unit, line, re.IGNORECASE)

        if match:
            value = float(match.group(1))
            # Skip zero/dummy values
            if value == 0:
                return None
            # Assume um if no unit
            return value * 1000

        return None

    def get_cell_info(self, cell_name):
        """
        Get detailed information about a cell.

        Args:
            cell_name: Name of the cell

        Returns:
            Formatted string with cell information
        """
        if cell_name not in self.logic_cells:
            return f"Cell {cell_name} not found"

        cell = self.logic_cells[cell_name]

        info = []
        info.append(f"\n{'='*60}")
        info.append(f"Cell: {cell.name}")
        info.append(f"{'='*60}")
        info.append(f"Ports: {', '.join(cell.ports)}")
        info.append(f"\nTransistors ({len(cell.transistors)}):")

        for t in cell.transistors:
            info.append(f"  {t.name}: {t.type}")
            info.append(f"    D={t.drain}, G={t.gate}, S={t.source}, B={t.bulk}")
            info.append(f"    W={t.width}nm, L={t.length}nm")

        info.append(f"\nResistors ({len(cell.resistors)}):")
        for r in cell.resistors[:10]:  # Show first 10
            info.append(f"  {r.name}: {r.node1} -- {r.node2} ({r.value})")
        if len(cell.resistors) > 10:
            info.append(f"  ... and {len(cell.resistors) - 10} more")

        info.append(f"\nConnection nodes: {len(cell.connections)}")

        return '\n'.join(info)

    def get_transistor_connectivity(self, cell_name):
        """
        Get transistor connectivity information by resolving resistor connections.

        This resolves the M1:DRN, M2:SRC notation to actual connected nodes.

        Args:
            cell_name: Name of the cell

        Returns:
            dict: {transistor_name: {drain: [connected_nodes], gate: [...], source: [...]}}
        """
        if cell_name not in self.logic_cells:
            return None

        cell = self.logic_cells[cell_name]
        connectivity = {}

        for t in cell.transistors:
            mos_name = t.name.replace('X', '')  # XM1 -> M1

            # Find all nodes connected to this transistor's terminals
            drain_connections = self._find_connected_nodes(cell, f"{mos_name}:DRN")
            gate_connections = self._find_connected_nodes(cell, f"{mos_name}:GATE")
            source_connections = self._find_connected_nodes(cell, f"{mos_name}:SRC")

            connectivity[t.name] = {
                'type': t.type,
                'width': t.width,
                'drain': drain_connections,
                'gate': gate_connections,
                'source': source_connections
            }

        return connectivity

    def _find_connected_nodes(self, cell, terminal):
        """
        Find all nodes connected to a terminal using BFS.

        Args:
            cell: LogicCell
            terminal: Terminal name (e.g., "M1:DRN")

        Returns:
            set: All connected nodes
        """
        if terminal not in cell.connections:
            return {terminal}

        visited = set()
        queue = [terminal]

        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)

            if node in cell.connections:
                for neighbor in cell.connections[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)

        return visited


def main():
    """Test SPI parser with example file."""
    import sys

    if len(sys.argv) < 2:
        spi_file = "/home/tkdgn2907/Deepsets_test/MAML/Projects/cdl_files/tcbn28hpcplusbwp30p140_110a_lpe_typical_filtered.spi"
    else:
        spi_file = sys.argv[1]

    print(f"Parsing SPI file: {spi_file}")
    parser = SPIParser(spi_file, verbose=True)

    print(f"\nFound {len(parser.logic_cells)} logic cells:")

    # Show first few cells
    for i, (name, cell) in enumerate(parser.logic_cells.items()):
        if i >= 5:
            print(f"... and {len(parser.logic_cells) - 5} more cells")
            break

        print(f"\n{'-'*40}")
        print(f"Cell: {name}")
        print(f"  Ports: {cell.ports}")
        print(f"  Transistors: {len(cell.transistors)}")
        print(f"  Resistors: {len(cell.resistors)}")

        # Show transistor info
        for t in cell.transistors[:3]:
            print(f"    {t.name}: {t.type} W={t.width}nm")

    # Detailed view of one cell
    test_cell = 'ND2D1BWP30P140'
    if test_cell in parser.logic_cells:
        print(parser.get_cell_info(test_cell))

        print("\nTransistor Connectivity:")
        conn = parser.get_transistor_connectivity(test_cell)
        for mos, info in conn.items():
            print(f"  {mos} ({info['type']}):")
            print(f"    Drain connected to: {info['drain']}")
            print(f"    Gate connected to: {info['gate']}")
            print(f"    Source connected to: {info['source']}")


if __name__ == "__main__":
    main()

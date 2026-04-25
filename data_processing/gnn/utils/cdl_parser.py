#!/usr/bin/env python
# coding: utf-8

"""
CDL Parser - Minimal SPICE CDL File Parser
===========================================

Lightweight CDL parser that extracts only the essential information:
logic cell definitions with transistor connectivity.

This is a minimal version of topology_based_extractor.py that contains
only the CDL parsing functionality, without any path extraction logic.

Usage:
    from cdl_parser import CDLParser

    parser = CDLParser('path/to/file.cdl')
    cells = parser.logic_cells  # Dict of {cell_name: LogicCell}
"""

import re
from collections import namedtuple

# Data structures for parsed CDL data
Transistor = namedtuple('Transistor', ['name', 'drain', 'gate', 'source', 'bulk', 'type', 'width', 'length', 'nfin'])
LogicCell = namedtuple('LogicCell', ['name', 'ports', 'transistors'])


class CDLParser:
    """
    Minimal CDL file parser.

    Parses SPICE CDL files and extracts logic cell definitions with
    transistor information. Only parsing, no path extraction or analysis.

    Attributes:
        logic_cells: Dictionary of parsed logic cells {cell_name: LogicCell}
    """

    def __init__(self, cdl_file_path):
        """
        Parse CDL file and extract logic cells.

        Args:
            cdl_file_path: Path to SPICE CDL file
        """
        self.spice_file_path = cdl_file_path
        self.logic_cells = {}
        self._parse_all_cells()

    def _parse_all_cells(self):
        """Parse all logic cells from CDL file."""
        with open(self.spice_file_path, 'r') as f:
            content = f.read()

        # Pattern: .SUBCKT cell_name ports ... .ENDS
        subckt_pattern = r'\.SUBCKT\s+(\w+)\s+([^\n]+)\n(.*?)\.ENDS'
        matches = re.findall(subckt_pattern, content, re.DOTALL | re.IGNORECASE)

        for cell_name, ports_str, body in matches:
            # Only parse logic cells (skip other subcircuits)
            if not self._is_logic_cell(cell_name):
                continue

            # Parse ports and transistors
            ports = ports_str.strip().split()
            transistors = self._parse_transistors(body)

            # Store parsed cell
            self.logic_cells[cell_name] = LogicCell(
                name=cell_name,
                ports=ports,
                transistors=transistors
            )

        print(f"   Parsed {len(self.logic_cells)} logic cells from CDL")

    def _is_logic_cell(self, cell_name):
        """
        Check if subcircuit is a logic cell.

        Args:
            cell_name: Subcircuit name

        Returns:
            True if logic cell, False otherwise
        """
        # Basic logic gate keywords (simple substring match)
        basic_keywords = [
            'INV', 'NAND', 'AND', 'OR', 'XOR', 'NOR', 'XNOR',
            'MUX', 'BUF', 'HA', 'FA', 'MAJ'
        ]

        # Complex AO/OA patterns with regex
        # Matches: AO21, AO211, A2O1A1Ix, A2O1A1O1Ix, O2A1O1Ix, etc.
        ao_oa_patterns = [
            r'AO\d+',        # AO21, AO211, AO22, etc.
            r'OA\d+',        # OA21, OA211, OA22, etc.
            r'AOI\d+',       # AOI21, AOI211, AOI22, etc.
            r'OAI\d+',       # OAI21, OAI211, OAI22, etc.
            r'A\d+O\d+',     # A2O1, A3O2, etc.
            r'O\d+A\d+',     # O2A1, O3A2, etc.
            r'[AO]\d+[AO]\d+([AO]\d+)*I?x?',  # A2O1A1Ix, A2O1A1O1Ix, O2A1O1Ix, etc. (repeating pattern)
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

        Args:
            circuit_body: Text body of subcircuit definition

        Returns:
            List of Transistor namedtuples
        """
        transistors = []

        # Pattern: name drain gate source bulk type w=width l=length nfin=nfin
        mos_pattern = r'(\w+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+w=(\S+)\s+l=(\S+)\s+nfin=(\d+)'

        for line in circuit_body.split('\n'):
            line = line.strip()

            # Skip empty lines, comments, and directives
            if not line or line.startswith('.') or line.startswith('*'):
                continue

            match = re.match(mos_pattern, line)
            if match:
                name, drain, gate, source, bulk, mos_type = match.groups()[:6]
                width_str, length_str, nfin_str = match.groups()[6:9]

                # Parse dimensions
                width = self._parse_dimension(width_str)
                length = self._parse_dimension(length_str)
                nfin = int(nfin_str)

                transistors.append(Transistor(
                    name=name,
                    drain=drain,
                    gate=gate,
                    source=source,
                    bulk=bulk,
                    type=mos_type,
                    width=width,
                    length=length,
                    nfin=nfin
                ))

        return transistors

    def _parse_dimension(self, dim_str):
        """
        Parse dimension string to float (in nm).

        Args:
            dim_str: Dimension string (e.g., "27n", "0.027u")

        Returns:
            Dimension value in nanometers
        """
        if dim_str.endswith('n'):
            return float(dim_str[:-1])
        elif dim_str.endswith('u'):
            return float(dim_str[:-1]) * 1000
        else:
            return float(dim_str)


# Backward compatibility alias
TopologyBasedExtractor = CDLParser

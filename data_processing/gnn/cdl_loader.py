#!/usr/bin/env python
# coding: utf-8

"""
Simple CDL Loader
=================

Minimal CDL file loader that provides only the essential functionality:
loading SPICE cell information from CDL files.

This replaces the bloated UnifiedLogicGraphTransformer with a simple,
focused loader that does one thing: parse CDL and provide all_logic_cells.

Usage:
    from cdl_loader import CDLLoader

    # Load single CDL file
    loader = CDLLoader('path/to/file.cdl')
    spice_cell = loader.all_logic_cells[cell_name]

    # Load and merge multiple CDL files
    loader = CDLLoader('file1.cdl')
    loader.merge_cdl('file2.cdl')
    loader.merge_cdl('file3.cdl')
"""

from cdl_parser import CDLParser


class CDLLoader:
    """
    Simple CDL file loader.

    This class wraps CDLParser to provide a clean interface
    for loading SPICE CDL files.

    Attributes:
        all_logic_cells: Dictionary of parsed SPICE cells {cell_name: cell_object}
        spice_file_path: Path to the primary CDL file
    """

    def __init__(self, cdl_file_path):
        """
        Load CDL file and parse SPICE cells.

        Args:
            cdl_file_path: Path to CDL file containing SPICE cell definitions
        """
        # Parse CDL file using CDLParser
        parser = CDLParser(cdl_file_path)

        # Store parsed cells (direct reference, not copy)
        self.all_logic_cells = parser.logic_cells

        # Store file path for backward compatibility
        self.spice_file_path = cdl_file_path

        print(f"   ✓ Loaded {len(self.all_logic_cells)} cells from {cdl_file_path}")

    def merge_cdl(self, cdl_file_path):
        """
        Load additional CDL file and merge cells into all_logic_cells.

        This is useful when you need to load cells from multiple CDL files
        (e.g., L, R, SL, SRAM variants).

        Args:
            cdl_file_path: Path to additional CDL file

        Returns:
            Number of new cells added
        """
        # Load additional CDL file
        parser = CDLParser(cdl_file_path)

        # Count new cells
        before_count = len(self.all_logic_cells)

        # Merge into existing dictionary
        self.all_logic_cells.update(parser.logic_cells)

        # Count after merge
        after_count = len(self.all_logic_cells)
        new_cells = after_count - before_count

        print(f"   ✓ Merged {len(parser.logic_cells)} cells from {cdl_file_path} "
              f"({new_cells} new)")

        return new_cells

    def get_cell(self, cell_name):
        """
        Get SPICE cell by name.

        Args:
            cell_name: Name of the cell to retrieve

        Returns:
            Cell object if found, None otherwise
        """
        return self.all_logic_cells.get(cell_name)

    def has_cell(self, cell_name):
        """
        Check if cell exists.

        Args:
            cell_name: Name of the cell to check

        Returns:
            True if cell exists, False otherwise
        """
        return cell_name in self.all_logic_cells

    def list_cells(self, pattern=None):
        """
        List all cell names, optionally filtered by pattern.

        Args:
            pattern: Optional string pattern to filter cell names

        Returns:
            List of cell names
        """
        if pattern is None:
            return list(self.all_logic_cells.keys())
        else:
            return [name for name in self.all_logic_cells.keys()
                    if pattern.upper() in name.upper()]


# For backward compatibility with existing code
UnifiedLogicGraphTransformer = CDLLoader

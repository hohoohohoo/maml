#!/usr/bin/env python
# coding: utf-8

"""
Simple SPI Loader
=================

Minimal TSMC SPI file loader that provides the essential functionality:
loading SPICE cell information from SPI files.

This is the TSMC equivalent of cdl_loader.py for ASAP7.

Usage:
    from spi_loader import SPILoader

    # Load single SPI file
    loader = SPILoader('path/to/file.spi')
    spice_cell = loader.all_logic_cells[cell_name]

    # Load and merge multiple SPI files
    loader = SPILoader('file1.spi')
    loader.merge_spi('file2.spi')
"""

from .spi_parser import SPIParser


class SPILoader:
    """
    Simple TSMC SPI file loader.

    This class wraps SPIParser to provide a clean interface
    for loading TSMC SPICE SPI files.

    Attributes:
        all_logic_cells: Dictionary of parsed SPICE cells {cell_name: cell_object}
        spi_file_path: Path to the primary SPI file
    """

    def __init__(self, spi_file_path, verbose=False):
        """
        Load SPI file and parse SPICE cells.

        Args:
            spi_file_path: Path to SPI file containing SPICE cell definitions
            verbose: Print detailed parsing info
        """
        # Parse SPI file using SPIParser
        self.parser = SPIParser(spi_file_path, verbose=verbose)

        # Store parsed cells (direct reference, not copy)
        self.all_logic_cells = self.parser.logic_cells

        # Store file path for backward compatibility
        self.spi_file_path = spi_file_path

        print(f"   ✓ Loaded {len(self.all_logic_cells)} cells from {spi_file_path}")

    def merge_spi(self, spi_file_path, verbose=False):
        """
        Load additional SPI file and merge cells into all_logic_cells.

        Args:
            spi_file_path: Path to additional SPI file
            verbose: Print detailed parsing info

        Returns:
            Number of new cells added
        """
        # Load additional SPI file
        parser = SPIParser(spi_file_path, verbose=verbose)

        # Count new cells
        before_count = len(self.all_logic_cells)

        # Merge into existing dictionary
        self.all_logic_cells.update(parser.logic_cells)

        # Count after merge
        after_count = len(self.all_logic_cells)
        new_cells = after_count - before_count

        print(f"   ✓ Merged {len(parser.logic_cells)} cells from {spi_file_path} "
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

    def get_cell_info(self, cell_name):
        """
        Get detailed information about a cell.

        Args:
            cell_name: Name of the cell

        Returns:
            Formatted string with cell information
        """
        return self.parser.get_cell_info(cell_name)

    def get_transistor_connectivity(self, cell_name):
        """
        Get transistor connectivity information by resolving resistor connections.

        Args:
            cell_name: Name of the cell

        Returns:
            dict: {transistor_name: {drain: [connected_nodes], gate: [...], source: [...]}}
        """
        return self.parser.get_transistor_connectivity(cell_name)


def main():
    """Test SPI loader with example file."""
    import sys

    if len(sys.argv) < 2:
        spi_file = "/home/tkdgn2907/Deepsets_test/MAML/Projects/cdl_files/tcbn28hpcplusbwp30p140_110a_lpe_typical_filtered.spi"
    else:
        spi_file = sys.argv[1]

    print(f"\n{'='*60}")
    print("SPI Loader Test")
    print(f"{'='*60}")

    # Load SPI file
    loader = SPILoader(spi_file, verbose=False)

    print(f"\nLoaded {len(loader.all_logic_cells)} cells")

    # List some cells
    print("\nSample cells:")
    for name in loader.list_cells()[:10]:
        cell = loader.get_cell(name)
        print(f"  {name}: {len(cell.transistors)} transistors, {len(cell.resistors)} resistors")

    # Filter cells by pattern
    print("\nNAND cells:")
    nand_cells = loader.list_cells('ND')
    for name in nand_cells[:5]:
        print(f"  {name}")
    if len(nand_cells) > 5:
        print(f"  ... and {len(nand_cells) - 5} more")

    # Detailed view
    test_cell = 'ND2D1BWP30P140'
    if loader.has_cell(test_cell):
        print(f"\n{loader.get_cell_info(test_cell)}")

        print("\nTransistor Connectivity:")
        conn = loader.get_transistor_connectivity(test_cell)
        if conn:
            for mos, info in conn.items():
                print(f"  {mos} ({info['type']}, W={info['width']}nm):")
                # Filter out internal nodes for cleaner display
                drain_external = [n for n in info['drain'] if not n.startswith('N_') and ':' not in n]
                gate_external = [n for n in info['gate'] if not n.startswith('N_') and ':' not in n]
                source_external = [n for n in info['source'] if not n.startswith('N_') and ':' not in n]
                print(f"    Drain -> {drain_external if drain_external else info['drain']}")
                print(f"    Gate -> {gate_external if gate_external else info['gate']}")
                print(f"    Source -> {source_external if source_external else info['source']}")


if __name__ == "__main__":
    main()

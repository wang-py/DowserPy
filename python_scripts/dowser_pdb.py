"""
dowser_pdb.py

Fixed-column PDB reading and writing shared by the pipeline stages.

Every file in this pipeline is parsed by column, never by whitespace splitting:
the coordinate fields run together as soon as a coordinate needs four digits
left of the decimal point, which happens routinely in these structures.

Column layout used throughout (1-based, as in the PDB spec):

    record     1-6      ATOM / HETATM
    serial     7-11     wrapped at 99999, see note below
    atom name  13-16
    resName    18-20
    chainID    22
    resSeq     23-26    cluster id after allwaters has clustered
    x, y, z    31-54
    occupancy  55-60
    bFactor    61-66    dowser energy once placeWat has run
    element    77-78

The serial and resSeq fields are five and four columns wide. Writing a value
too large for the field pushes every later column right by one and silently
corrupts the file for every fixed-column reader downstream, so both are wrapped
on write. Nothing in this pipeline keys on the serial number alone.
"""

import os
import sys

MAX_SERIAL = 99999
MAX_RESSEQ = 9999


class Water(object):
    """A single water oxygen record."""

    __slots__ = ("serial", "resseq", "x", "y", "z", "occupancy", "bfactor")

    def __init__(self, serial, resseq, x, y, z, occupancy=1.0, bfactor=0.0):
        self.serial = serial
        self.resseq = resseq
        self.x = x
        self.y = y
        self.z = z
        self.occupancy = occupancy
        self.bfactor = bfactor

    @property
    def coord(self):
        return (self.x, self.y, self.z)

    def key(self):
        """Identity used to join waters across pipeline stages."""
        return (self.serial, self.resseq)


def read_waters(path, atom_names=("O", "OW", "OH2"), resnames=("HOH", "WAT"),
                warn=True):
    """
    Read water oxygen records from a PDB file.

    Records whose atom name or residue name fall outside the given sets are
    skipped, as are records whose numeric fields do not parse.

    Skipping is reported on stderr unless `warn` is false. A file written by a
    tool that overflowed its serial or residue field has every later column
    shifted, so those records fail the residue-name test and vanish here; the
    count and the residue names that were rejected are the quickest way to see
    that rather than silently working on part of the file.
    """
    waters = []
    skipped = {}

    def note(reason):
        skipped[reason] = skipped.get(reason, 0) + 1

    with open(path, "r") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            if line[12:16].strip() not in atom_names:
                note("atom name %r" % line[12:16])
                continue
            if line[17:20].strip() not in resnames:
                note("residue name %r" % line[17:20])
                continue
            try:
                waters.append(
                    Water(
                        serial=int(line[6:11]),
                        resseq=int(line[22:26]),
                        x=float(line[30:38]),
                        y=float(line[38:46]),
                        z=float(line[46:54]),
                        occupancy=float(line[54:60] or 1.0),
                        bfactor=float(line[60:66] or 0.0),
                    )
                )
            except ValueError:
                note("unparsable coordinate or number fields")
                continue

    if warn and skipped:
        total = sum(skipped.values())
        detail = ", ".join(
            "%d with %s" % (count, reason)
            for reason, count in sorted(skipped.items(),
                                        key=lambda item: -item[1])[:3]
        )
        sys.stderr.write(
            "Warning: %s: skipped %d of %d atom records (%s).\n"
            "         A file whose serial or residue field overflowed has "
            "every later column shifted by one, which looks exactly like "
            "this; regenerate it if the count is unexpected.\n"
            % (os.path.basename(path), total, total + len(waters), detail)
        )

    return waters


def format_water(serial, resseq, x, y, z, bfactor, atom_name="OW", chain=" "):
    """
    Format one water oxygen as a fixed-column HETATM record.

    `serial` and `resseq` are wrapped to fit their fields; see the module
    docstring for why that matters.
    """
    safe_serial = ((int(serial) - 1) % MAX_SERIAL) + 1
    safe_resseq = ((int(resseq) - 1) % MAX_RESSEQ) + 1
    # Atom name occupies columns 13-16; single-letter names are conventionally
    # indented by one so the element letter sits in column 14.
    name = (" %-3s" % atom_name)[:4]

    return (
        "HETATM%5d %-4s %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s\n"
        % (
            safe_serial,
            name,
            "HOH",
            chain,
            safe_resseq,
            x,
            y,
            z,
            1.00,
            bfactor,
            "O",
        )
    )


def write_waters(path, waters, remarks=(), atom_name="OW"):
    """
    Write waters as a PDB file, renumbering serials sequentially from 1 and
    preserving each water's resseq (cluster id) and bfactor (dowser energy).
    """
    with open(path, "w") as handle:
        for remark in remarks:
            handle.write("REMARK    %s\n" % remark)
        for index, water in enumerate(waters):
            handle.write(
                format_water(
                    index + 1,
                    water.resseq,
                    water.x,
                    water.y,
                    water.z,
                    water.bfactor,
                    atom_name=atom_name,
                )
            )
        handle.write("END\n")


def read_remarks(path):
    """Return the REMARK text carried by a PDB file, without the keyword."""
    remarks = []
    with open(path, "r") as handle:
        for line in handle:
            if line.startswith("REMARK"):
                remarks.append(line[6:].strip())
            elif line.startswith(("ATOM", "HETATM")):
                break
    return remarks

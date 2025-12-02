import sys
import struct
import os

def patch_elf(path):
    try:
        with open(path, 'r+b') as f:
            # Read ELF header
            e_ident = f.read(16)
            if not e_ident.startswith(b'\x7fELF'):
                return
            
            # Check 64-bit (EI_CLASS == 2)
            if e_ident[4] != 2:
                # Skip non-64-bit files for now (IPEX is likely 64-bit)
                return

            # Read e_phoff (Program header offset) at offset 32 for Elf64
            f.seek(32)
            e_phoff = struct.unpack('<Q', f.read(8))[0]
            
            # Read e_phentsize (offset 54) and e_phnum (offset 56)
            f.seek(54)
            e_phentsize = struct.unpack('<H', f.read(2))[0]
            e_phnum = struct.unpack('<H', f.read(2))[0]

            # Iterate program headers
            PT_GNU_STACK = 0x6474e551
            
            for i in range(e_phnum):
                offset = e_phoff + i * e_phentsize
                f.seek(offset)
                # Read p_type (4 bytes)
                p_type = struct.unpack('<I', f.read(4))[0]
                
                if p_type == PT_GNU_STACK:
                    # Found it. Flags are at offset +4 in Elf64_Phdr
                    # struct Elf64_Phdr {
                    #   uint32_t p_type;
                    #   uint32_t p_flags;
                    #   ...
                    # }
                    f.seek(offset + 4)
                    p_flags = struct.unpack('<I', f.read(4))[0]
                    
                    if p_flags & 0x1: # PF_X (Executable)
                        print(f"Patching {path}: Clearing executable stack flag")
                        new_flags = p_flags & ~0x1
                        f.seek(offset + 4)
                        f.write(struct.pack('<I', new_flags))
                    return

            # If we reach here, no PT_GNU_STACK found. 
            # We could add one, but usually it implies executable stack if missing? 
            # Or defaults to non-executable on modern systems?
            # For now, just ignore.
            
    except Exception as e:
        print(f"Error processing {path}: {e}")

if __name__ == "__main__":
    for path in sys.argv[1:]:
        patch_elf(path)

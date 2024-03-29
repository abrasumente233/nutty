#!/usr/bin/env python3

from enum import Enum, auto

mem = [0] * 1024 * 1024 # 1 MiB of memory
x = [0] * 32 # registers
pc = 0

class Instr(Enum):
    lui = auto()
    auipc = auto()
    jal = auto()
    jalr = auto()
    beq = auto()
    bne = auto()
    blt = auto()
    bge = auto()
    bltu = auto()
    bgeu = auto()
    lb = auto()
    lh = auto()
    lw = auto()
    lbu = auto()
    lhu = auto()
    sb = auto()
    sh = auto()
    sw = auto()
    addi = auto()
    slti = auto()
    sltiu = auto()
    xori = auto()
    ori = auto()
    andi = auto()
    slli = auto()
    srli = auto()
    srai = auto()
    add = auto()
    sub = auto()
    sll = auto()
    slt = auto()
    sltu = auto()
    xor = auto()
    srl = auto()
    sra = auto()
    bitwise_or = auto()
    bitwise_and = auto()
    fence = auto()
    fence_i = auto()
    ecall = auto()
    ebreak = auto()
    csrrw = auto()
    csrrs = auto()
    csrrc = auto()
    csrrwi = auto()
    csrrsi = auto()
    csrrci = auto()

class Opcode(Enum):
    Lui    = 0b0110111
    Auipc  = 0b0010111
    Jal    = 0b1101111
    Jalr   = 0b1100111
    Branch = 0b1100011
    Load   = 0b0000011
    Store  = 0b0100011
    OpImm  = 0b0010011
    Op     = 0b0110011
    Fence  = 0b0001111
    System = 0b1110011

opcode_format_map = {
    Opcode.Lui:    'U',
    Opcode.Auipc:  'U',
    Opcode.Jal:    'J',
    Opcode.Jalr:   'I',
    Opcode.Branch: 'B',
    Opcode.Load:   'I',
    Opcode.Store:  'S',
    Opcode.OpImm:  'I',
    Opcode.Op:     'R',
    Opcode.Fence:  'I',
    Opcode.System: 'I',
}

# map hell, someone please help me.
# problem: 
#   1. given opcode, need fmt
#   2. given opcode and funct, need instr
# wrong answer only! (ah! SQL!)
funct_instr_map = {
    (Opcode.Lui,    None):    Instr.lui,
    (Opcode.Auipc,  None):    Instr.auipc,
    (Opcode.Jal,    None):    Instr.jal,
    (Opcode.Jalr,   0b000):   Instr.jalr,

    (Opcode.Branch, 0b000):   Instr.beq,
    (Opcode.Branch, 0b001):   Instr.bne,
    (Opcode.Branch, 0b100):   Instr.blt,
    (Opcode.Branch, 0b101):   Instr.bge,
    (Opcode.Branch, 0b110):   Instr.bltu,
    (Opcode.Branch, 0b111):   Instr.bgeu,

    (Opcode.Load,   0b000):   Instr.lb,
    (Opcode.Load,   0b001):   Instr.lh,
    (Opcode.Load,   0b010):   Instr.lw,
    (Opcode.Load,   0b100):   Instr.lbu,
    (Opcode.Load,   0b101):   Instr.lhu,

    (Opcode.Store,  0b000):   Instr.sb,
    (Opcode.Store,  0b001):   Instr.sh,
    (Opcode.Store,  0b010):   Instr.sw,

    (Opcode.OpImm,  0b000):   Instr.addi,
    (Opcode.OpImm,  0b010):   Instr.slti,
    (Opcode.OpImm,  0b011):   Instr.sltiu,
    (Opcode.OpImm,  0b100):   Instr.xori,
    (Opcode.OpImm,  0b110):   Instr.ori,
    (Opcode.OpImm,  0b111):   Instr.andi,
    (Opcode.OpImm,  0b001):   Instr.slli,

    # FIXME: Assert upper 8 bits
    (Opcode.Op,     0b000):   Instr.add,
    (Opcode.Op,     0b000):   Instr.sub,
    (Opcode.Op,     0b001):   Instr.sll,
    (Opcode.Op,     0b010):   Instr.slt,
    (Opcode.Op,     0b011):   Instr.sltu,
    (Opcode.Op,     0b100):   Instr.xor,
    (Opcode.Op,     0b110):   Instr.bitwise_or,
    (Opcode.Op,     0b111):   Instr.bitwise_and,

    (Opcode.Fence,  0b000):   Instr.fence,
    (Opcode.Fence,  0b001):   Instr.fence_i,

    # FIXME: wtf, these two have same encoding
    #(Opcode.Env,    0b000):   Instr.ecall,
    #(Opcode.Env,    0b000):   Instr.ebreak,
    (Opcode.System, 0b001):   Instr.csrrw,
    (Opcode.System, 0b010):   Instr.csrrs,
    (Opcode.System, 0b011):   Instr.csrrc,
    (Opcode.System, 0b101):   Instr.csrrwi,
    (Opcode.System, 0b110):   Instr.csrrsi,
    (Opcode.System, 0b111):   Instr.csrrci,
}

instrs = (
    (Instr.lui,   'U', Opcode.Lui),
    (Instr.auipc, 'U', Opcode.Auipc),
    (Instr.jal,   'J', Opcode.Jal),
    (Instr.jalr,  'I', Opcode.Jalr),
    (Instr.beq,   'B', Opcode.Branch),
    (Instr.bne,   'B', Opcode.Branch),
    (Instr.blt,   'B', Opcode.Branch),
    (Instr.bge,   'B', Opcode.Branch),
    (Instr.bltu,  'B', Opcode.Branch),
    (Instr.bgeu,  'B', Opcode.Branch),
    (Instr.lb,    'I', Opcode.Load),
    (Instr.lh,    'I', Opcode.Load),
    (Instr.lw,    'I', Opcode.Load),
    (Instr.lbu,   'I', Opcode.Load),
    (Instr.lhu,   'I', Opcode.Load),
    (Instr.sb,    'S', Opcode.Store),
    (Instr.sh,    'S', Opcode.Store),
    (Instr.sw,    'S', Opcode.Store),
    (Instr.addi,  'I', Opcode.OpImm),
    (Instr.slti,  'I', Opcode.OpImm),
    (Instr.sltiu, 'I', Opcode.OpImm),
    (Instr.xori,  'I', Opcode.OpImm),
    (Instr.ori,   'I', Opcode.OpImm),
    (Instr.andi,  'I', Opcode.OpImm),
    (Instr.slli,  'I', Opcode.OpImm),
    (Instr.srli,  'I', Opcode.OpImm),
    (Instr.srai,  'I', Opcode.OpImm),
    (Instr.add,   'R', Opcode.Op),
    (Instr.sub,   'R', Opcode.Op),
    (Instr.sll,   'R', Opcode.Op),
    (Instr.slt,   'R', Opcode.Op),
    (Instr.sltu,  'R', Opcode.Op),
    (Instr.xor,   'R', Opcode.Op),
    (Instr.srl,   'R', Opcode.Op),
    (Instr.sra,   'R', Opcode.Op),
    (Instr.bitwise_or,   'R', Opcode.Op),
    (Instr.bitwise_and,  'R', Opcode.Op),
    (Instr.fence,   'I', Opcode.Fence),
    (Instr.fence_i, 'I', Opcode.Fence),
    (Instr.ecall,   'I', Opcode.System),
    (Instr.ebreak,  'I', Opcode.System),
    (Instr.csrrw,   'I', Opcode.System),
    (Instr.csrrs,   'I', Opcode.System),
    (Instr.csrrc,   'I', Opcode.System),
    (Instr.csrrwi,  'I', Opcode.System),
    (Instr.csrrsi,  'I', Opcode.System),
    (Instr.csrrci,  'I', Opcode.System),
)

def sign_ext(x: int, signed: bool, sign_pos: int) -> int:
    for b in range(sign_pos, 32):
        x |= (signed << b)
    return x

def sign_ext_32(x: int, sign_pos: int) -> int:
    if ((x >> sign_pos) & 1) != 0:
        for b in range(sign_pos + 1, 32):
            x |= (1 << b)

    return x

# extract bits
def ex(bits: int, high: int, low: int) -> int:
    bits = bits >> low
    length = high - low + 1
    mask = (1 << length) - 1
    return bits & mask

# extract inplace: extract bits and shift right `to_low` bits
def exip(bits: int, high: int, low: int, to_low: int) -> int:
    return ex(bits, high, low) << to_low

def to_signed(x: int) -> int:
    signed = x >> 31
    x &= 0x7fffffff
    return x - signed * 0x80000000

# FIXME: rs1, rs2, rd, funct can be meaningless for certain
#        instruction format.
def decode(I: int):

    opcode = Opcode(ex(I, 6, 0))
    rs1 = ex(I, 19, 15)
    rs2 = ex(I, 24, 20)
    rd  = ex(I, 11, 7)

    if opcode in (Opcode.Lui, Opcode.Auipc, Opcode.Jal):
        funct = None
    else:
        funct = ex(I, 14, 12)

    # decode immediate
    fmt = opcode_format_map[opcode]
    sign_bit = (I >> 31) == 1
    imm = None
    if fmt == 'I':
        imm = sign_ext(exip(I, 30, 20, 0), sign_bit, 11)
    elif fmt == 'S':
        imm = sign_ext(exip(I, 11, 7, 0) | exip(I, 30, 25, 5), sign_bit, 11)
    elif fmt == 'B':
        imm = sign_ext(exip(I, 7, 7, 11) | exip(I, 30, 25, 5) | exip(I, 11, 8, 1), sign_bit, 12)
    elif fmt == 'U':
        imm = I & 0xfffff000
    elif fmt == 'J':
        imm = sign_ext(exip(I, 30, 21, 1) | exip(I, 20, 20, 11) | exip(I, 19, 12, 12), sign_bit, 20)
    elif fmt == 'R':
        imm = None
    else:
        raise Exception("unreachable")

    instr = None
    upper = ex(I, 31, 26)
    if opcode == Opcode.OpImm and funct == 0b101:
        if upper == 0b000000:
            instr = Instr.srli
        elif upper == 0b010000:
            instr = Instr.srai
            imm &= 0x1f # type: ignore
        else:
            assert False
    elif opcode == Opcode.Op and funct == 0b000:
        if upper == 0b000000:
            instr = Instr.add
        elif upper == 0b010000:
            instr = Instr.sub
        else:
            assert False
    elif opcode == Opcode.Op and funct == 0b101:
        if upper == 0b000000:
            instr = Instr.srl
        elif upper == 0b010000:
            instr = Instr.sra
        else:
            assert False
    elif opcode == Opcode.Op and funct == 0b000:
        upper = ex(I, 31, 20)
        if upper == 1:
            instr = Instr.ebreak
        elif upper == 0:
            instr = Instr.ecall
        else:
            assert False
    else:
        instr = funct_instr_map[(opcode, funct)] # type: ignore

    print('0x{:08x}\t{}'.format(pc, str(instr)[6:]), end='\t')
    if fmt in ('R', 'I', 'U', 'J'):
        print('r{}\t'.format(rd), end='')
    if fmt in ('R', 'I', 'S', 'B'):
        print('r{}\t'.format(rs1), end='')
    if fmt in ('R', 'S', 'B'):
        print('r{}\t'.format(rs2), end='')
    if fmt != 'R':
        print('0x{:x}'.format(imm))

    return (instr, opcode, imm, rs1, rs2, rd)

def trunc(x: int) -> int:
    return x & 0xffffffff

# x is an n-bit number to be shifted m times
def sra(x,n,m):
    if x & 2**(n-1) != 0:  # MSB is 1, i.e. x is negative
        filler = int('1'*m + '0'*(n-m),2)
        x = (x >> m) | filler  # fill in 0's with 1's
        return x
    else:
        return x >> m

# FIXME: make sure x0 is always 0
# interpret an instruction
def interpret_inst() -> bool:
    global pc

    # instruction fetch
    I = mem[pc] | (mem[pc + 1] << 8) | (mem[pc + 2] << 16) | \
        (mem[pc + 3] << 24)

    # stop running when we hit zero instruction
    if I == 0:
        return False

    # instruction decode
    (instr, opcode, imm, rs1, rs2, rd) = decode(I)

    # nah, skip system opcodes
    if opcode == Opcode.System:
        pc += 4
        return True

    # instruction execute
    changed_pc = False

    if instr == Instr.lui:
        x[rd] = imm
    elif instr == Instr.auipc:
        x[rd] = trunc(pc + imm)
    elif instr == Instr.jal:
        x[rd] = pc + 4
        pc = trunc(pc + imm) # TODO: make sure imm is right
        changed_pc = True
    elif instr == Instr.jalr:
        t = pc + 4
        pc = trunc(x[rs1] + imm) & ~1
        changed_pc = True
        x[rd] = t
    elif instr == Instr.beq:
        if x[rs1] == x[rs2]:
            pc = trunc(pc + imm)
            changed_pc = True
    elif instr == Instr.bne:
        if x[rs1] != x[rs2]:
            pc = trunc(pc + imm)
            changed_pc = True
    elif instr == Instr.blt:
        if to_signed(x[rs1]) < to_signed(x[rs2]):
            pc = trunc(pc + imm)
            changed_pc = True
    elif instr == Instr.bge:
        if to_signed(x[rs1]) >= to_signed(x[rs2]):
            pc = trunc(pc + imm)
            changed_pc = True
    elif instr == Instr.bltu:
        if x[rs1] < x[rs2]:
            pc = trunc(pc + imm)
            changed_pc = True
    elif instr == Instr.bgeu:
        if x[rs1] >= x[rs2]:
            pc = trunc(pc + imm)
            changed_pc = True
    elif instr == Instr.lb:
        x[rd] = sign_ext_32(mem[trunc(x[rs1] + imm)], 7)
    elif instr == Instr.lh:
        off = trunc(x[rs1] + imm)
        data = mem[off] | (mem[off + 1] << 8) # little endian
        x[rd] = sign_ext_32(data, 15)
    elif instr == Instr.lw:
        off = trunc(x[rs1] + imm)
        data = mem[off] | (mem[off + 1] << 8) | (mem[off + 2] << 16) | (mem[off + 3] << 24)
        x[rd] = data
    elif instr == Instr.lbu:
        x[rd] = mem[trunc(x[rs1] + imm)]
    elif instr == Instr.lhu:
        off = trunc(x[rs1] + imm)
        data = mem[off] | (mem[off + 1] << 8) # little endian
        x[rd] = data
    elif instr == Instr.sb:
        off = trunc(x[rs1] + imm)
        data = x[rs2] & 0xff
        mem[off] = data
    elif instr == Instr.sh:
        off = trunc(x[rs1] + imm)
        data = x[rs2] & 0xff
        mem[off]     =  x[rs2]       & 0xff
        mem[off + 1] = (x[rs2] >> 8) & 0xff
    elif instr == Instr.sw:
        off = trunc(x[rs1] + imm)
        data = x[rs2] & 0xff
        mem[off]     =  x[rs2]        & 0xff
        mem[off + 1] = (x[rs2] >> 8)  & 0xff
        mem[off + 2] = (x[rs2] >> 16) & 0xff
        mem[off + 3] = (x[rs2] >> 24) & 0xff
    elif instr == Instr.addi:
        x[rd] = trunc(x[rs1] + imm)
    elif instr == Instr.slti:
        x[rd] = 1 if to_signed(x[rs1]) < to_signed(imm) else 0
    elif instr == Instr.sltiu:
        x[rd] = 1 if x[rs1] < imm else 0
    elif instr == Instr.xori:
        x[rd] = x[rs1] ^ imm
    elif instr == Instr.ori:
        x[rd] = x[rs1] | imm
    elif instr == Instr.andi:
        x[rd] = x[rs1] & imm
    elif instr == Instr.slli:
        x[rd] = trunc(x[rs1] << imm)
    elif instr == Instr.srli:
        x[rd] = x[rs1] >> imm
    elif instr == Instr.srai:
        x[rd] = sra(x[rs1], 32, imm)
    elif instr == Instr.add:
        x[rd] = trunc(x[rs1] + x[rs2])
    elif instr == Instr.sub:
        assert(False and "can we sub directly?")
    elif instr == Instr.sll:
        x[rd] = trunc(x[rs1] << x[rs2])
    elif instr == Instr.slt:
        x[rd] = 1 if to_signed(x[rs1]) < to_signed(x[rs2]) else 0
    elif instr == Instr.sltu:
        x[rd] = 1 if x[rs1] < x[rs2] else 0
    elif instr == Instr.xor:
        x[rd] = x[rs1] ^ x[rs2]
    elif instr == Instr.srl:
        x[rd] = x[rs1] >> x[rs2]
    elif instr == Instr.sra:
        x[rd] = sra(x[rs1], 32, x[rs2])
    elif instr == Instr.bitwise_or:
        x[rd] = x[rs1] | x[rs2]
    elif instr == Instr.bitwise_and:
        x[rd] = x[rs1] & x[rs2]
    else:
        raise Exception("Illegal instruction")

    if not changed_pc:
        pc += 4

    return True
    
def load_prog(filename: str):
    p = 0
    with open(filename, "rb") as f:
        byte = f.read(1)
        while byte:
            mem[p] = int.from_bytes(byte, "little")
            p += 1
            byte = f.read(1)

def dump():
    print('{: <3}: 0x{:08x}  '.format('pc', pc))
    for i in range(32):
        print('{: <3}: 0x{:08x}  '.format('r' + str(i), x[i]), end='')
        if (i - 3) % 4 == 0:
            print()

def run():
    max_step = 100
    while interpret_inst():
        max_step -= 1
        if max_step == 0:
            break

if __name__ == '__main__':
    load_prog("simple.bin")

    run()
    dump()

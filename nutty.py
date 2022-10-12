#!/usr/bin/env python3

from enum import Enum, auto

mem = [0] * 1024 # 1 KiB of memory
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

class Opcode(Enum):
    Lui    = 0b0110111
    Auipc  = 0b0010111
    Jal    = 0b1101111
    Jalr   = 0b1101111
    Branch = 0b1101111
    Load   = 0b0000011
    Store  = 0b0100011
    OpImm  = 0b0010011
    Op     = 0b0110011
    Fence  = 0b0001111
    Env    = 0b1110011

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
    Opcode.Env:    'I',
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
    #(Opcode.OpImm,  0b101):   Instr.srli,
    #(Opcode.OpImm,  0b101):   Instr.srai, # FIXME!!: determined by upper bits too!

    # FIXME: Assert upper 8 bits
    (Opcode.Op,     0b000):   Instr.add,
    (Opcode.Op,     0b000):   Instr.sub,
    (Opcode.Op,     0b001):   Instr.sll,
    (Opcode.Op,     0b010):   Instr.slt,
    (Opcode.Op,     0b011):   Instr.sltu,
    (Opcode.Op,     0b100):   Instr.xor,
    #(Opcode.Op,     0b101):   Instr.srl,
    #(Opcode.Op,     0b101):   Instr.sra,
    (Opcode.Op,     0b110):   Instr.bitwise_or,
    (Opcode.Op,     0b111):   Instr.bitwise_and,

    (Opcode.Fence,  0b000):   Instr.fence,
    (Opcode.Fence,  0b001):   Instr.fence_i,

    # FIXME: wtf, these two have same encoding
    #(Opcode.Env,    0b000):   Instr.ecall,
    #(Opcode.Env,    0b000):   Instr.ebreak,
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
    (Instr.ecall,   'I', Opcode.Env),
    (Instr.ebreak,  'I', Opcode.Env),
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

# extract bits and shift right `to_low` bits
def ex(bits: int, high: int, low: int, to_low: int):
    bits = bits >> low
    length = high - low + 1
    mask = sum([2**i for i in range(length)])
    bits &= mask
    return bits << to_low

def to_signed(x: int) -> int:
    signed = x >> 31
    x &= 0x7fffffff
    return x - signed * 0x80000000

# FIXME: rs1, rs2, rd, funct can be meaningless for certain
#        instruction format.
def decode(I: int):

    opcode = Opcode(I & 0b1111111)
    rs1 = (I >> 15) & 0b11111
    rs2 = (I >> 20) & 0b11111
    rd = (I >> 7) & 0b11111

    if opcode in (Opcode.Lui, Opcode.Auipc, Opcode.Jal):
        funct = None
    else:
        funct = (I >> 12) & 0b111 # FIXME: U and J doesn't have funct

    # decode immediate
    fmt = opcode_format_map[opcode]
    sign_bit = (I >> 31) == 1
    imm = None
    if fmt == 'I':
        imm = sign_ext(ex(I, 30, 20, 0), sign_bit, 11)
    elif fmt == 'S':
        imm = sign_ext(ex(I, 11, 7, 0) | ex(I, 30, 25, 5), sign_bit, 11)
    elif fmt == 'B':
        imm = sign_ext(ex(I, 7, 7, 11) | ex(I, 30, 25, 5) | ex(I, 11, 8, 1), sign_bit, 12)
    elif fmt == 'U':
        imm = I & 0xfffff000
    elif fmt == 'J':
        imm = sign_ext(ex(I, 30, 21, 1) | ex(I, 20, 20, 11) | ex(I, 19, 12, 12), sign_bit, 20)
    elif fmt == 'R':
        imm = None
    else:
        raise Exception("unreachable")

    instr = None
    upper = I >> 24
    if opcode == Opcode.OpImm and funct == 0b101:
        if upper == 0b00000000:
            instr = Instr.srli
        elif upper == 0b01000000:
            instr = Instr.srai
            imm &= 0x1f
    elif opcode == Opcode.Op and funct == 0b000:
        if upper == 0b00000000:
            instr = Instr.add
        elif upper == 0b01000000:
            instr = Instr.sub
    elif opcode == Opcode.Op and funct == 0b101:
        if upper == 0b00000000:
            instr = Instr.srl
        elif upper == 0b01000000:
            instr = Instr.sra
    else:
        instr = funct_instr_map[(opcode, funct)]

    print("{}, (r{}, r{}, {}) -> r{}".format(instr, rs1, rs2, imm, rd))

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


    # instruction execute
    changed_pc = False

    if instr == Instr.lui:
        x[rd] = imm
    elif instr == Instr.auipc:
        x[rd] = pc + imm
    elif instr == Instr.jal:
        x[rd] = pc + 4
        pc += imm # TODO: make sure imm is right
        changed_pc = True
    elif instr == Instr.jalr:
        t = pc + 4
        pc = (x[rs1] + imm) & ~1
        changed_pc = True
        x[rd] = t
    elif instr == Instr.beq:
        if x[rs1] == x[rs2]:
            pc += imm
            changed_pc = True
    elif instr == Instr.bne:
        if x[rs1] != x[rs2]:
            pc += imm
            changed_pc = True
    elif instr == Instr.blt:
        if to_signed(x[rs1]) < to_signed(x[rs2]):
            pc += imm
            changed_pc = True
    elif instr == Instr.bge:
        if to_signed(x[rs1]) >= to_signed(x[rs2]):
            pc += imm
            changed_pc = True
    elif instr == Instr.bltu:
        if x[rs1] < x[rs2]:
            pc += imm
            changed_pc = True
    elif instr == Instr.bgeu:
        if x[rs1] >= x[rs2]:
            pc += imm
            changed_pc = True
    elif instr == Instr.lb:
        x[rd] = sign_ext_32(mem[x[rs1] + imm], 7)
    elif instr == Instr.lh:
        off = x[rs1] + imm
        data = mem[off] | (mem[off + 1] << 8) # little endian
        x[rd] = sign_ext_32(data, 15)
    elif instr == Instr.lw:
        off = x[rs1] + imm
        data = mem[off] | (mem[off + 1] << 8) | (mem[off + 2] << 16) | (mem[off + 3] << 24)
        x[rd] = data
    elif instr == Instr.lbu:
        x[rd] = mem[x[rs1] + imm]
    elif instr == Instr.lhu:
        off = x[rs1] + imm
        data = mem[off] | (mem[off + 1] << 8) # little endian
        x[rd] = data
    elif instr == Instr.sb:
        off = x[rs1] + imm
        data = x[rs2] & 0xff
        mem[off] = data
    elif instr == Instr.sh:
        off = x[rs1] + imm
        data = x[rs2] & 0xff
        mem[off]     =  x[rs2]       & 0xff
        mem[off + 1] = (x[rs2] >> 8) & 0xff
    elif instr == Instr.sw:
        off = x[rs1] + imm
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

def run():
    while interpret_inst():
        pass
    print('Execution ended')

if __name__ == '__main__':
    load_prog("simple.bin")

    run()
    print(x[8])

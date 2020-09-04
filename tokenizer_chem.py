# Copyright (c) 2018, University of Cambridge, UK and IBM Research GmbH, Switzerland All rights reserved.
#   Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens), 'smiles: {}, token: {}'.format(smi, tokens)
    return ' '.join(tokens)

def smi_tokenizer_molecule(smi):
    return smi.replace('.', ' ').replace('>', ' > ')

def smi_tokenizer_rp_atom_c_molecule(s):
    a, b, c = s.split('>')
    a = smi_tokenizer(a)
    c = smi_tokenizer(c)
    b = smi_tokenizer_molecule(b)
    return ' > '.join([a,b,c])

if __name__ == "__main__":
    s = 'CCC.CC>CCCCCC.C>CCCCCCCC'
    assert smi_tokenizer_molecule(s) == 'CCC CC > CCCCCC C > CCCCCCCC'
    assert smi_tokenizer(s) == 'C C C . C C > C C C C C C . C > C C C C C C C C'
    assert smi_tokenizer_rp_atom_c_molecule(s) == 'C C C . C C > CCCCCC C > C C C C C C C C'

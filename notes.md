# Project Notes
## Container Management
### Launching zeek
docker run -t -v /home/Projects/NPDP:/home/files:z zeekurity/zeek

### Launching thatDot (lab)
`docker run --security-opt seccomp=unconfined -p 8080:8080 thatdot/novelty:0.12.0`
Remove --security and seccomp if not in lab, was not neccsarry on normal computer

## Novelty Detector Transformations
### Add a tag to thatdot transformation
`that => ({"observation": [that[0], that[1]], "tag": that})`

### UNSW Transformation - Selective
`that => ({"observation":[that[5], that[4], that[13], that[0], that[12], that[2], that[17], that[16], that[29], that[28], that[22], that[23], that[8], that[7], that[3], that[20], that[6], that[15], that[14], that[1]], "tag":[that[47], that[48]]})`
Results did not detect attack values, almost all "anomalous" scores were not labled as attack observations.

### UNSW Transformation - Inclusive
`that => ({"observation":[that[39], that[38], that[35], that[19], that[18], that[10], that[24], that[45], that[37], that[36], that[5], that[4], that[13], that[9], that[46], that[11], that[41], that[25], that[44], that[40], that[42], that[43], that[0], that[12], that[2], that[17], that[16], that[29], that[28], that[22], that[23], that[8], that[7], that[3], that[34], that[33], that[32], that[20], that[21], that[27], that[26], that[31], that[30], that[6], that[15], that[14], that[1]], "tag":[that[47], that[48]]})`

### UNSW Transform - Discretized

`that => ({"observation":[that['is_ftp_login'], that['ct_ftp_cmd'], that['dwin'], that['is_sm_ips_ports'], that['trans_depth'], that['swin'], that['dttl'], that['sttl'], that['state'], that['ct_flw_http_mthd'], that['ct_dst_sport_ltm'], that['ct_state_ttl'], that['ct_src_dport_ltm'], that['service'], that['ct_src_ltm'], that['ct_dst_ltm'], that['ct_srv_dst'], that['ct_srv_src'], that['ct_dst_src_ltm'], that['dloss'], that['sloss'], that['dpkts'], that['spkts'], that['proto'], that['response_body_len'], that['dmean'], that['smean'], that['dbytes'], that['sbytes'], that['synack'], that['ackdat'], that['tcprtt'], that['djit'], that['stcpb'], that['dtcpb'], that['dinpkt'], that['dload'], that['sjit'], that['sinpkt'], that['rate'], that['dur'], that['sload']], "tag":[that['label'], that['attack_cat'], that['id']]})`

### UNSW Transformation - Selective
`that => ({"observation":[that['is_ftp_login'], that['ct_ftp_cmd'], that['dwin'], that['is_sm_ips_ports'], that['label'], that['trans_depth'], that['swin'], that['dttl'], that['sttl'], that['state'], that['ct_flw_http_mthd'], that['ct_dst_sport_ltm'], that['ct_state_ttl'], that['ct_src_dport_ltm'], that['attack_cat'], that['service'], that['ct_src_ltm'], that['ct_dst_ltm'], that['ct_srv_dst'], that['ct_srv_src'], that['ct_dst_src_ltm'], that['dloss'], that['sloss'], that['dpkts'], that['spkts'], that['proto'], that['response_body_len'], that['dmean'], that['smean'], that['dbytes'], that['sbytes'], that['synack'], that['ackdat'], that['tcprtt'], that['djit'], that['stcpb'], that['dtcpb'], that['dinpkt'], that['dload'], that['sjit'], that['sinpkt'], that['rate'], that['dur'], that['sload'], that['id'] ], "tag":[that[47], that[48]]})`

### Transformation for Binned Top Ten Features
`that => ({
  "observation":[    that['dur'], 
    that['spkts'], 
    that['dpkts'], 
    that['sbytes'], 
    that['dbytes'], 
    that['rate'], 
    that['sttl'], 
    that['dttl'], 
    that['sload'], 
    that['dload']
  ],
  "tag":[    that['attack_cat'], 
    that['label']
  ]
})`

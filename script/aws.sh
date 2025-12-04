aws_instance_ip="ec2-18-116-13-180.us-east-2.compute.amazonaws.com"
aws_pem="~/Documents/Code/arturo-aws.pem"

action=$1

if [ "$action" == "push" ]; then
    scp -i $aws_pem -r src/* ubuntu@$aws_instance_ip:~/air/arturo/agent-mt/src/
    scp -i $aws_pem config.env ubuntu@$aws_instance_ip:~/air/arturo/agent-mt/
    scp -i $aws_pem requirements.txt ubuntu@$aws_instance_ip:~/air/arturo/agent-mt/
fi

if [ "$action" == "pull" ]; then
    scp -i $aws_pem ubuntu@$aws_instance_ip:~/air/arturo/agent-mt/data/benchmark_results/* data/benchmark_results/
fi

if [ "$action" == "data" ]; then
    scp -i $aws_pem -r data/raw/wmt25-terminology-track2 ubuntu@$aws_instance_ip:~/air/arturo/agent-mt/data/
fi
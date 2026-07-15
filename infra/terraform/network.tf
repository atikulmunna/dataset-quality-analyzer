resource "aws_vpc" "worker" {
  cidr_block           = "10.42.0.0/24"
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags                 = { Name = "${local.name}-worker" }
}

resource "aws_internet_gateway" "worker" {
  vpc_id = aws_vpc.worker.id
  tags   = { Name = "${local.name}-worker" }
}

resource "aws_subnet" "worker" {
  count                   = 2
  vpc_id                  = aws_vpc.worker.id
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  cidr_block              = cidrsubnet(aws_vpc.worker.cidr_block, 2, count.index)
  map_public_ip_on_launch = false
  tags                    = { Name = "${local.name}-worker-${count.index + 1}" }
}

resource "aws_route_table" "worker" {
  vpc_id = aws_vpc.worker.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.worker.id
  }
  tags = { Name = "${local.name}-worker" }
}

resource "aws_route_table_association" "worker" {
  count          = 2
  subnet_id      = aws_subnet.worker[count.index].id
  route_table_id = aws_route_table.worker.id
}

resource "aws_security_group" "worker" {
  name        = "${local.name}-worker-egress"
  description = "No ingress; audit workers initiate HTTPS only"
  vpc_id      = aws_vpc.worker.id

  egress {
    description = "AWS HTTPS APIs and ECR"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}


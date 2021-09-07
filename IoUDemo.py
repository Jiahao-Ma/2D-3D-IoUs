"""
Visualize the process of calculating rotated IoU step by step.
"""
import numpy as np
import torch
from cuda_op.cuda_ext import sort_v
EPSLION = 1e-8
# TODO: Convert the box (x, y, w, h, alpha) to 4corners (x1,y1,x2,y2,x3,y3,x4,y4).
def boxes2corners(boxes:torch.Tensor):
    """
        Args: 
            box: `torch.Tensor` shape: (B, N, 5), contains (x, y, w, h, alpha) of batch
        Returns:
            corners: `torch.Tensor` shape: (B, N, 4, 2)
    """
    x = boxes[..., 0:1] # (B, N, 1)
    y = boxes[..., 1:2]
    w = boxes[..., 2:3]
    h = boxes[..., 3:4]
    alpha = boxes[..., 4::5]
    tx = torch.Tensor([0.5, -0.5, -0.5, 0.5]).unsqueeze(0).unsqueeze(0).to(boxes.device)
    tx = tx * w
    ty = torch.Tensor([0.5, 0.5, -0.5, -0.5]).unsqueeze(0).unsqueeze(0).to(boxes.device)
    ty = ty * h
    # (B, N, 4, 2)
    txty = torch.stack([tx, ty], dim=-1)
    cos = torch.cos(alpha)
    sin = torch.sin(alpha)
    # (2, 2) counter clockwise
    rotate = torch.Tensor([[cos, sin],
                           [-sin, cos]]).to(device=boxes.device)
    # rotate
    txty = txty @ rotate
    # (B, N, 4, 2)
    xy = torch.cat([x, y], dim=-1).unsqueeze(2).repeat([1,1,4,1])
    # translate
    corners = xy + txty
    return corners

# TODO: Find the intersection of boxes.
def boxes_intersection(corners1:torch.Tensor, corners2:torch.Tensor):
    """
        Args:
            corners1: `torch.Tensor` shape: (B, N, 4, 2), contains the l t r b corners of boxes
            corners2: `torch.Tensor` shape: (B, N, 4, 2), contains the l t r b corners of boxes
        Returns: 
            inters: `torch.Tensor` shape: (B, N, 16, 2) intersections of 2 boxes. There are 4 combinations for one line,
                    each box has 4 line. Thus, there are 16 (4 x 4) combinations finnally. 
            mask: `torch.Tensor` [!BOOL!] shape: (B, N, 16) mask of intersection. The mask marks the valid intersection.
        
        How to get the intersection of two lines: 
            Reference: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    """
    # construct lines (B, N, 4, 4)
    lines1 = torch.cat([corners1, corners1[:, :, [1, 2, 3, 0], :]], dim=-1)
    lines2 = torch.cat([corners2, corners2[:, :, [1, 2, 3, 0], :]], dim=-1)
    # (B, N, 4, 4, 4)
    ext_lines1 = lines1.unsqueeze(3).repeat([1,1,1,4,1])
    ext_lines2 = lines2.unsqueeze(2).repeat([1,1,4,1,1])
    # (B, N, 4, 4)
    x1 = ext_lines1[..., 0]
    y1 = ext_lines1[..., 1]
    x2 = ext_lines1[..., 2]
    y2 = ext_lines1[..., 3]
    
    x3 = ext_lines2[..., 0]
    y3 = ext_lines2[..., 1]
    x4 = ext_lines2[..., 2]
    y4 = ext_lines2[..., 3]

    # Refer to intersection of lines: 
    # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    molecular_t = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    molecular_u = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)

    t = molecular_t / denominator
    u = molecular_u / denominator

    mask_t = (t > 0.) * (t < 1.)
    mask_u = (u > 0.) * (u < 1.)
    mask = mask_t * mask_u
    t = molecular_t / ( denominator + EPSLION)
    inters = torch.stack([ x1 + t * (x2 - x1), y1 + t * (y2 - y1)], dim=-1)
    inters = inters * mask.float().unsqueeze(-1)
    # (B, N, 4, 4, 2), (B, N, 4, 4) -> (B, N, 16, 2), (B, N, 16) 
    B = inters.size()[0]
    N = inters.size()[1]
    return inters.reshape((B, N, -1, 2)), mask.reshape((B, N, -1))

#TODO: Judge if box is inside another box.
def box1_in_box2(corners1:torch.Tensor, corners2:torch.Tensor):
    """
        Args:
            corners1: `torch.Tensor` shape: (B, N, 4, 2), contains the l t r b corners of boxes
            corners2: `torch.Tensor` shape: (B, N, 4, 2), contains the l t r b corners of boxes
        Return:
            c: `torch.Tensor` shape: (B, N, 4), mark the corners that are inside another box
            
    """
    # (B, N, 1, 2)
    a = corners2[:, :, 0:1, :]
    b = corners2[:, :, 1:2, :]
    d = corners2[:, :, 3:4, :]
    ab = b - a
    ad = d - a
    # (B, N, 4, 2)
    am = corners1 - a
    # calculate the projection of am on ab through dot product 
    projected_ab = torch.sum(ab * am, dim=-1)
    projected_ad = torch.sum(ad * am, dim=-1)
    # calculate the length of ab and ad
    norm_ab = torch.sum( ab * ab, dim=-1)
    norm_ad = torch.sum( ad * ad, dim=-1)
    # line ab is parrel to  x axis, ad is parrel to y axis
    # judge if these corners' x/y coordinate are inside another box
    c_x = ( projected_ab / norm_ab > - 1e-6) * ( projected_ab / norm_ab < 1. + 1e-6)
    c_y = ( projected_ad / norm_ad > - 1e-6) * ( projected_ad / norm_ad < 1. + 1e-6)
    c = c_x * c_y
    return c

# TODO: Build vertices.
def build_vertices(corners1:torch.Tensor, corners2:torch.Tensor, inters:torch.Tensor, c1_in_2:torch.Tensor, c2_in_1:torch.Tensor, mask_inter:torch.Tensor):
    """
        Args:
            corners1: `torch.Tensor` shape: (B, N, 4, 2), contains the l t r b corners of boxes
            corners2: `torch.Tensor` shape: (B, N, 4, 2), contains the l t r b corners of boxes
            inters: `torch.Tensor` shape: (B, N, 16, 2) intersections of 2 boxes. 
            mask_inter: `torch.Tensor` shape: (B, N, 16) mask of intersection. The mask marks the valid intersection.
            c1_in_2, c2_in_1: `torch.Tensor` shape: (B, N, 4), mark the corners that are inside another box
        Returns:
            vertices: `torch.Tensor` shape: (B, N, 24, 2) [24 = 4+4+16] concanate corners1, corners2 and inters
            masks: `torch.Tensor` shape: (B, N, 24) concanate the c1_in_2, c2_in_1 and mask_inter. It marks the valid corner of the overlap (polygon)
    """
    # vertices: (B, N, 4+4+16, 2)
    vertices = torch.cat([corners1, corners2, inters], dim=2)
    # masks: (B, N, 4+4+16)
    masks = torch.cat([c1_in_2, c2_in_1, mask_inter], dim=2)
    return vertices, masks

#TODO: Sort vertices.
def sort_vertices(vertices, masks):
    """
        Args:
            vertices: `torch.Tensor` shape: (B, N, 24, 2) [24 = 4+4+16] concanate corners1, corners2 and inters
            masks: `torch.Tensor` shape: (B, N, 24) concanate the c1_in_2, c2_in_1 and mask_inter. It marks the valid corner of the overlap (polygon)
        Return:
            sorted_index: `torch.Tensor` shape: (B, N, 24)
        
        Steps:(1)normalize (- mean) (2) sort
    """
    # (B, N)
    num_valid = torch.sum(masks.int(), dim=-1).int()
    # (B, N, 1, 2)
    mean = torch.sum( vertices * masks.unsqueeze(-1), dim=2, keepdim=True) / num_valid.unsqueeze(dim=-1).unsqueeze(dim=-1)
    vertices = vertices - mean
    # NOTICE: masks must be bool instead of int or float
    return sort_v(vertices, masks, num_valid).long()

#TODO: Calculate the are of overlap.
def calculate_area(vertices, sorted_index):
    """
        Args:
            vertices: `torch.Tensor` shape: (B, N, 24, 2) [24 = 4+4+16] concanate corners1, corners2 and inters
            sorted_index: `torch.Tensor` shape: (B, N, k) sorted index of vertices. `k` is the number of valid corners of overlap
        Returns:
            area: `torch.Tensor` (B, N) area of overlap
    """
    sorted_index = sorted_index.unsqueeze(-1).repeat([1,1,1,2])
    # (B, N, k, 2)
    selected = torch.gather(vertices, 2, sorted_index)
    # How to calculate the area of polygon? 
    # Refer to shoelace formula: https://en.wikipedia.org/wiki/Shoelace_formula
    # (B, N, k-1)
    total = selected[:, :, 0:-1, 0] * selected[:, :, 1:, 1] - selected[:, :, 0:-1, 1] * selected[:, :, 1:, 0]
    total = torch.sum(total, dim=2)
    area = torch.abs(total) / 2
    return area, selected

#TODO: Calculate Ious
def IoUs(bbox1, bbox2, overlap):
    """
        Args:
            bbox1, bbox2: `torch.Tensor` shape: (B, N, 5), contains (x, y, w, h, alpha) of batch
            overlap: `torch.Tensor` shape(B, N) the area of overlap
        Returns:
            ious: `torch.Tensor` shape(B, N)
    """
    area1 = bbox1[..., 2] * bbox1[..., 3]
    area2 = bbox2[..., 2] * bbox2[..., 3]
    return overlap / (area1 + area2 - overlap)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    def viz(box:np.ndarray, ax=None, color='green', linewidth=3, xrange=(-3,3), yrange=(-3,3)):
        colors = ['red', 'green', 'blue', 'black']
        # box: (4, 2)
        if ax == None:
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(111)
            # plt.xlim(xrange)
            # plt.ylim(yrange)
        for i in range(len(box)):
            if i != len(box) - 1:
                plt.plot([box[i][0], box[i+1][0]], [box[i][1], box[i+1][1]], color=colors[i], linewidth=linewidth)
            else:
                plt.plot([box[i][0], box[0][0]], [box[i][1], box[0][1]], color=colors[i], linewidth=linewidth)
        return ax
    
    def viz_scatter(inters, mask, ax, color='red'):
        # inters: (B, N, 16, 2) mask: (B, N, 16)  default B, N = 1, 1
        assert ax != None, 'ax must not be empty'
        inters = inters * mask.unsqueeze(dim=-1) # repeated operation. Removing it is ok.
        inters = inters.reshape(-1, 2)
        mask = mask.reshape(-1)
        for idx, m in enumerate(mask):
            if int(m) == 0:
                continue
            inter = inters.cpu().numpy()[idx]
            ax.scatter(inter[0], inter[1], color=color)
        return ax
    
    def evaluate():
        from IoU import IoUs2D
        device = torch.device('cuda:0')
        box_0 = torch.tensor([.0, .0, 2., 2., .0], device=device) # x y w h alpha

        def test_same_box():
            expect = 1.
            box_1 = torch.tensor([.0, .0, 2., 2., .0], device=device)
            result = IoUs2D(box_0[None, None, ...], box_1[None, None, ...])[0].cpu().numpy()
            print("expect:", expect, "get:", result[0,0])

        def test_same_edge():
            expect = 0.
            box_1 = torch.tensor([.0, 2, 2., 2., .0], device=device)
            result = IoUs2D(box_0[None, None, ...], box_1[None, None, ...])[0].cpu().numpy()
            print("expect:", expect, "get:", result[0,0])

        def test_same_edge_offset():
            expect = 0.3333
            box_1 = torch.tensor([.0, 1., 2., 2., .0], device=device)
            result = IoUs2D(box_0[None, None, ...], box_1[None, None, ...])[0].cpu().numpy()
            print("expect:", expect, "get:", result[0,0])

        def test_same_box2():
            expect = 1
            box_1 = torch.tensor([38, 120, 1.3, 20, (50 * np.pi) / 180], device=device)
            result = IoUs2D(box_1[None, None, ...], box_1[None, None, ...])[0].cpu().numpy()
            print("expect:", expect, "get:", result[0,0])
        
        test_same_box()
        test_same_edge()
        test_same_edge_offset()
        test_same_box2()

    def demo():
        device = torch.device('cuda')
        alpha = 120
        box1 = torch.tensor([0, 0, 2, 2, 0], device=device).unsqueeze(0).unsqueeze(0)
        box2 = torch.tensor([0, 0, 2, 2, (alpha * np.pi)/180], device=device).unsqueeze(0).unsqueeze(0)
        
        corners1 = boxes2corners(box1)
        corners2 = boxes2corners(box2)
        ax = viz(corners1.reshape(-1, 2))
        ax = viz(corners2.reshape(-1, 2), ax, color='red')
        # plt.show()
        inters, mask_inters = boxes_intersection(corners1, corners2)
        ax = viz_scatter(inters, mask_inters, ax)

        c1_in_2 = box1_in_box2(corners1, corners2)
        c2_in_1 = box1_in_box2(corners2, corners1)

        vertices, masks = build_vertices(corners1, corners2, inters, c1_in_2, c2_in_1, mask_inters)

        sorted_index = sort_vertices(vertices, masks)
        
        overlap, _ = calculate_area(vertices, sorted_index)

        print('IoUs:', IoUs(box1, box2, overlap))
        
        # visualize 
        plt.show()

    #-------------#
    #     Demo    #
    #-------------#
    
    # visualize the process of calculating rotated IoU step by step.
    demo()

    # evaluate different pairs of boxes
    # evaluate()
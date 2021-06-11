import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import matplotlib.patches as mpatches


class class_visualisation:
    def plot_contour(self, ax):
        if self.data['fine_area'] == 'rectangle':
            ax.plot(self.contour_points[:, 0], self.contour_points[:, 1], color = "black")

            ax.set_xlim([self.contour_points[0, 0], 5 * self.contour_points[1, 0] / 4])
            ax.set_ylim([self.contour_points[1, 1], 5 * self.contour_points[2, 1] / 4])

        elif self.data['fine_area'] == 'thick_walled_cylinder':
            inner_circle = mpatches.Arc([0, 0], self.inner_radius * 2, self.inner_radius * 2, angle = 0, theta1 = 0, theta2 = 90)
            outer_circle = mpatches.Arc([0, 0], self.outer_radius * 2, self.outer_radius * 2, angle = 0, theta1 = 0, theta2 = 90)

            ax.add_patch(inner_circle)
            ax.add_patch(outer_circle)

            ax.plot(self.contour_points[:2, 0], self.contour_points[:2, 1], color = "k")
            ax.plot(self.contour_points[2:-1, 0], self.contour_points[2:-1, 1], color = "k")

            ax.set_xlim([0, self.outer_radius * 9 / 8])
            ax.set_ylim([0, self.outer_radius * 9 / 8])

        elif self.data['fine_area'] == 'bearing':
            ball_center_point = self.inner_radius + (self.outer_radius - self.inner_radius) / 2.0
            ball_radius = 0.2926354830241924
            dist = ball_radius - (self.outer_radius - self.inner_radius) / 4.0

            inner_ring_hole = mpatches.Arc([0, 0], self.inner_radius * 2, self.inner_radius * 2, angle = 0, theta1 = 0, theta2 = 90)
            inner_ring = mpatches.Arc([0, 0], (self.inner_radius + (self.outer_radius - self.inner_radius) / 4 - dist) * 2, (self.inner_radius + (self.outer_radius - self.inner_radius) / 4 - dist) * 2, angle = 0, theta1 = 0, theta2 = 90)

            outer_ring_hole = mpatches.Arc([0, 0], self.outer_radius * 2, self.outer_radius * 2, angle = 0, theta1 = 0, theta2 = 90)
            outer_ring = mpatches.Arc([0, 0], (self.inner_radius + 3 * (self.outer_radius - self.inner_radius) / 4 + dist) * 2, (self.inner_radius + 3 * (self.outer_radius - self.inner_radius) / 4 + dist) * 2, angle = 0, theta1 = 0, theta2 = 90)
            
            for i in range(5):
                coef1 = 0
                coef2 = 360
                if i == 0:
                    coef2 = 180
                elif i == 4:
                    coef1 = 270
                    coef2 = 90
                degree = i * (np.pi / 8)
                ring = mpatches.Arc([ball_center_point * np.cos(degree), ball_center_point * np.sin(degree)], ball_radius * 2, ball_radius * 2, theta1 = coef1, theta2 = coef2)
                ax.add_patch(ring)

            ax.add_patch(inner_ring_hole)
            ax.add_patch(inner_ring)
            ax.add_patch(outer_ring_hole)
            ax.add_patch(outer_ring)

            ax.plot(self.contour_points[:2, 0], self.contour_points[:2, 1], color = "k")
            ax.plot(self.contour_points[2:-1, 0], self.contour_points[2:-1, 1], color = "k")

            ax.set_xlim([0, self.outer_radius * 9 / 8])
            ax.set_ylim([0, self.outer_radius * 9 / 8])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xlabel('$x$', fontsize = 20)
        ax.set_ylabel('$y$', fontsize = 20)
        return ax


    def plot_area_diagram(self, save = False):
        if not os.path.exists(f'results/{self.data["fine_area"]}/{self.data["task"]}/core'):
            os.makedirs(f'results/{self.data["fine_area"]}/{self.data["task"]}/core')

        fig, ax = plt.subplots()
        ax.plot(self.area_points_coords[:, 0], self.area_points_coords[:, 1], 'o', color = 'black', alpha = 0.4, markersize = 2)
        ax.triplot(self.area_points_coords[:, 0], self.area_points_coords[:, 1], self.area_elements, color = 'grey', alpha = 0.5)

        ax = self.plot_contour(ax)

        if self.data['fine_area'] == 'rectangle':
            fig.set_figwidth(8)
            fig.set_figheight(5)
        else:
            fig.set_figwidth(8)
            fig.set_figheight(8)

        if save:
            route = f'results/{self.data["fine_area"]}/{self.data["task"]}/core/area_diagram.png'
            plt.savefig(route)
        else:
            plt.show()


    def plot_area_decomposition(self, save = False):
        if not os.path.exists(f'results/{self.data["fine_area"]}/{self.data["task"]}/core'):
            os.makedirs(f'results/{self.data["fine_area"]}/{self.data["task"]}/core')

        fig, ax = plt.subplots(2, 1)

        for idx, value in self.dict_subd_elements.items():
            ax[idx].triplot(self.area_points_coords[:, 0], self.area_points_coords[:, 1], self.area_elements[value], color = 'grey', alpha = 0.5)

            ax[idx] = self.plot_contour(ax[idx])

            ax[idx].set_xlabel('$x$', fontsize = 20)
            ax[idx].set_ylabel('$y$', fontsize = 20)
            ax[idx].set_xticklabels([])
            ax[idx].set_yticklabels([])
        
        fig.set_figwidth(5)
        fig.set_figheight(10)

        if save:
            route = f'results/{self.data["fine_area"]}/{self.data["task"]}/core/area_decomposition.png'
            plt.savefig(route)
        else:
            plt.show()


    def plot_area_coarse(self, save = False):
        if not os.path.exists(f'results/{self.data["fine_area"]}/{self.data["task"]}/core'):
            os.makedirs(f'results/{self.data["fine_area"]}/{self.data["task"]}/core')

        fig, ax = plt.subplots()
        ax.plot(self.area_points_coords[:, 0], self.area_points_coords[:, 1], 'o', color = 'black', alpha = 0.4, markersize = 2)
        ax.triplot(self.area_coarse_points_coords[:, 0], self.area_coarse_points_coords[:, 1], self.area_coarse_elements, color = 'grey', alpha = 0.5)

        ax = self.plot_contour(ax)

        ax.set_xlabel('$x$', fontsize = 20)
        ax.set_ylabel('$y$', fontsize = 20)
        ax.tick_params(labelsize = 14)

        fig.set_figwidth(9)
        fig.set_figheight(6)

        if save:
            route = f'results/{self.data["fine_area"]}/{self.data["task"]}/core/area_coarse_{self.data["coarse_area"]}.png'
            plt.savefig(route)
        else:
            plt.show()


    def plot_displacement_distribution(self, save = False):
        if not os.path.exists(f'results/{self.data["fine_area"]}/{self.data["task"]}/core'):
            os.makedirs(f'results/{self.data["fine_area"]}/{self.data["task"]}/core')

        fig, ax = plt.subplots()
        triang = mtri.Triangulation(self.area_points_coords[:, 0], self.area_points_coords[:, 1], self.area_elements)
        
        if self.data['fine_area'] == 'rectangle':
            ax_plt = ax.tricontourf(triang, self.u[:, 1] * self.coef_u)
            cbar = plt.colorbar(ax_plt)
            cbar.set_label(f'$u_y, м$', fontsize = 20)
            fig.set_figwidth(9)
            fig.set_figheight(6)
        else:
            ax_plt = ax.tricontourf(triang, self.u_polar[:, 0] * self.coef_u)
            cbar = plt.colorbar(ax_plt)
            cbar.set_label(f'$u_r, м$', fontsize = 20)
            fig.set_figwidth(11)
            fig.set_figheight(9)

        ax = self.plot_contour(ax)
        cbar.ax.tick_params(labelsize = 14)
        ax.tick_params(labelsize = 14)

        if save:
            route = f'results/{self.data["fine_area"]}/{self.data["task"]}/core/displacement_distribution.png'
            plt.savefig(route)
        else:
            plt.show()


    def plot_displacement_polar(self):
        plt.polar(self.u_polar[self.inner_radius_points, 1], self.u_polar[self.inner_radius_points, 0], 'o')
        plt.polar(self.u_polar[self.outer_radius_points, 1], self.u_polar[self.outer_radius_points, 0], 'o')
        plt.show()


    def plot_pressure_distribution(self, save):
        if not os.path.exists(f'results/{self.data["fine_area"]}/{self.data["task"]}/core'):
            os.makedirs(f'results/{self.data["fine_area"]}/{self.data["task"]}/core')

        fig, ax = plt.subplots()
        triang = mtri.Triangulation(self.area_points_coords[:, 0], self.area_points_coords[:, 1], self.area_elements)

        if self.data['fine_area'] == 'rectangle':
            type = 1
            ax_plt = ax.tricontourf(triang, self.sigma_points[:, type])
            name = "_x" if type == 0 else "_y"
            special_name = name
            fig.set_figwidth(9)
            fig.set_figheight(6)
        else:
            type = 1
            ax_plt = ax.tricontourf(triang, self.sigma_points_polar[:, type] * self.coef_sigma)
            name = "_r" if type == 0 else "_phi"
            special_name = "_r" if type == 0 else "_{phi}"
            fig.set_figwidth(11)
            fig.set_figheight(9)

        cbar = plt.colorbar(ax_plt)
        cbar.set_label(f'$\sigma{special_name}$, МПа', fontsize = 20)
        cbar.ax.tick_params(labelsize = 14)
        ax = self.plot_contour(ax)
        ax.tick_params(labelsize = 14)
        if save:
            route = f'results/{self.data["fine_area"]}/{self.data["task"]}/core/pressure_distribution{name}.png'
            plt.savefig(route)
        else:
            plt.show()


    def internal_plot_displacement(self, point_coords, elements, special_points = None, draw_points = False):
        fig, ax = plt.subplots()

        ax.triplot(point_coords[:, 0], point_coords[:, 1], elements)
        if draw_points:
            ax.plot(point_coords[special_points, 0], point_coords[special_points, 1], 'o')

        ax = self.plot_contour(ax)

        fig.set_figwidth(10)
        fig.set_figheight(10)

        plt.show()


    def internal_plot_displacement_coarse(self, points_coords, coarse_element, elements):
        fig, ax = plt.subplots()
        ax.plot(points_coords[:, 0], points_coords[:, 1], 'o')
        ax.triplot(coarse_element[:, 0], coarse_element[:, 1], elements.copy())

        ax = self.plot_contour(ax)

        fig.set_figwidth(10)
        fig.set_figheight(10)
        fig.set_facecolor('mintcream')

        plt.show()


if __name__ == "__main__":
    pass